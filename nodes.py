import json
import os
import requests
import sys
import logging
from tqdm import tqdm
import re
import folder_paths
from pathlib import Path
import importlib.util
from nodes import LoraLoader, UNETLoader, CheckpointLoaderSimple, VAELoader, CLIPLoader

LOG_PREFIX = "[ComfyUI-OnDemand-Loaders]"


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers = []
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"{LOG_PREFIX} %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"Starting dynamic import of nodes.py from ComfyUI-GGUF...")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
module_path = Path(parent_dir) / 'ComfyUI-GGUF' / '__init__.py'
module_name = "ComfyUI-GGUF"
module_gguf = None
if module_path.exists():
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        module_gguf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module_gguf 
        spec.loader.exec_module(module_gguf)

        logger.info(f"Successfully found and imported UnetLoaderGGUF dynamically.")
    except AttributeError as ae:
        logger.error(f"'UnetLoaderGGUF' not found. Check the class name: {ae}")
        module_gguf = None
    except Exception as e:
        logger.error(f"Error during module execution (nodes.py content error): {e}")
        module_gguf = None

else:
    # Handle the missing file case gracefully
    logger.warning(f"ComfyUI-GGUF installation not found! Expected location: {module_path.as_posix().parent}")
    logger.warning(f"OnDemand GGUF Loaders will not be available")


# Function to load configuration 
def load_config(config_filename="config.json"):
    
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)
    
    default_config = { 
        "loras": [
            {
                "name": "Lora n1",
                "url": "not_valid_url",
            },
            {
                "name": "Lora n2",
                "url": "not_valid_url"
            }
        ]
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_filename}")

        # add None to all lists
        for key in config:
            if isinstance(config[key], list):
                none_entry = {"name": "None", "url": None}
                if none_entry not in config[key]:
                    config[key].insert(0, none_entry)

        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config

def _get_api_key_for_url(model_url, api_key_param):
    """
    Determines the API key to use based on the model_url.
    It checks for a provided api_key_param first, then environment variables.
    """
    if model_url.startswith("https://civitai.com"):
        return api_key_param or os.environ.get('CIVITAI_TOKEN')
    elif model_url.startswith("https://huggingface.co"):
        return api_key_param or os.environ.get('HUGGINGFACE_TOKEN')
    else:
        return api_key_param # Return provided key if URL doesn't match known platforms


def _download_model(model_url, model_name, destination_dir, api_key, download_chunks):
    """
    Handles the download of a model from a given URL to a specified directory.
    
    Args:
        model_url (str): The URL of the model to download.
        model_name (str): The name of the model (for logging purposes).
        destination_dir (str): The directory where the model should be saved.
        api_key (str): API key for authentication, if required.
        download_chunks (int): The size of download chunks in KB.
        
    Returns:
        str: The full path to the downloaded model file, or None if an error occurred.
    """
    os.makedirs(destination_dir, exist_ok=True)

    headers = None
    if api_key:
        logger.info(f"Using provided API key")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

    try:
        response = requests.get(model_url, stream=True, allow_redirects=True, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request for '{model_name}' from '{model_url}': {e}")
        return None

    model_filename = None
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
        if filename_match:
            model_filename = filename_match.group(1).strip()
    
    if not model_filename:
        # Fallback to extracting filename from URL if Content-Disposition is missing or malformed
        model_filename = os.path.basename(model_url)

    model_filepath = os.path.join(destination_dir, model_filename)

    if os.path.exists(model_filepath):
        logger.info(f"File '{model_filename}' already exists at '{model_filepath}'. Skipping download.")
        return model_filepath
    else:
        logger.info(f"Downloading '{model_name}' from '{model_url}' to '{model_filepath}'")
        try:
            total_size = int(response.headers.get('content-length', 0))
            block_size = download_chunks * 1024
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{LOG_PREFIX} Downloading {model_name}") as progress_bar:
                with open(model_filepath, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
            logger.info(f"Successfully downloaded '{model_name}' filename {model_filename}.")
            return model_filepath
        except Exception as e:
            logger.error(f"An unexpected error occurred during download of '{model_name}': {e}")
            return None

def _get_model_url_from_config(model_name, model_type_key):
    """
    Retrieves the URL for a given model name from the NODE_CONFIG.
    """
    model_url = None
    for model in NODE_CONFIG.get(model_type_key, []):
        if model["name"] == model_name:
            model_url = model["url"]
            break
    if not model_url:
        logger.error(f"Model URL not found for name: {model_name} in {model_type_key}")
    return model_url

NODE_CONFIG = load_config()

class OnDemandLoraLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        loras = [lora["name"] for lora in NODE_CONFIG.get("loras", []) ]
       
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "clip": ("CLIP", ),
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "download_lora"
    DESCRIPTION = "Load loras models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    CATEGORY = "loaders"

    def download_lora(self, model, lora_name, strength_model, strength_clip, clip=None, api_key=None, download_chunks=None):
        self.lora_loader = LoraLoader()

        destination_dir = os.path.join(folder_paths.models_dir, "loras")

        lora_url = _get_model_url_from_config(lora_name, "loras")
        if not lora_url:
            return model, clip # Return original model/clip if URL not found

        api_key = _get_api_key_for_url(lora_url, api_key)

        lora_filepath = _download_model(lora_url, lora_name, destination_dir, api_key, download_chunks)
        if not lora_filepath:
            return model, clip # Return original model/clip if download fails

        lora_filename = os.path.basename(lora_filepath)

        # Load the LORA using the existing LoraLoader
        model_lora, clip_lora = self.lora_loader.load_lora(model, clip, lora_filename, strength_model, strength_clip)
        return model_lora, clip_lora


class OnDemandUNETLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("diffusion_models", []) ]
       
        return {
            "required": {
                "unet_name": (models,),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1})
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download_unet"
    DESCRIPTION = "Load diffusion models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    CATEGORY = "loaders"

    def download_unet(self, unet_name, weight_dtype, api_key=None, download_chunks=None):
        self.unet_loader = UNETLoader()

        destination_dir = os.path.join(folder_paths.models_dir, "diffusion_models")

        model_url = _get_model_url_from_config(unet_name, "diffusion_models")
        if not model_url:
            return None

        api_key = _get_api_key_for_url(model_url, api_key)

        model_filepath = _download_model(model_url, unet_name, destination_dir, api_key, download_chunks)
        if not model_filepath:
            return None

        model_filename = os.path.basename(model_filepath)

        # Load the Model using the existing UNETLoader
        model_output = self.unet_loader.load_unet(model_filename, weight_dtype)
        return model_output


class OnDemandCheckpointLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("checkpoints", []) ]
       
        return {
            "required": {
                "ckpt_name": (models,)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "download_checkpoint"
    DESCRIPTION = "Load checkpoint models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"
    CATEGORY = "loaders"

    def download_checkpoint(self, ckpt_name, api_key=None, download_chunks=None):
        self.checkpoint_loader = CheckpointLoaderSimple()

        destination_dir = os.path.join(folder_paths.models_dir, "checkpoints")

        model_url = _get_model_url_from_config(ckpt_name, "checkpoints")
        if not model_url:
            return None

        api_key = _get_api_key_for_url(model_url, api_key)

        model_filepath = _download_model(model_url, ckpt_name, destination_dir, api_key, download_chunks)
        if not model_filepath:
            return None, None, None # Return None for all outputs if download fails

        model_filename = os.path.basename(model_filepath)

        # Load the checkpoint using the existing CheckpointLoaderSimple
        return self.checkpoint_loader.load_checkpoint(model_filename)

class OnDemandVAELoader:
    
    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("vae_models", []) ]
       
        return {
            "required": {
                "vae_name": (models,)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1})
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "download_vae"
    DESCRIPTION = "Load vae models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"
    CATEGORY = "loaders"

    def download_vae(self, vae_name, api_key=None, download_chunks=None):
        self.vae_loader = VAELoader()

        destination_dir = os.path.join(folder_paths.models_dir, "vae")

        model_url = _get_model_url_from_config(vae_name, "vae_models")
        if not model_url:
            return None

        api_key = _get_api_key_for_url(model_url, api_key)

        model_filepath = _download_model(model_url, vae_name, destination_dir, api_key, download_chunks)
        if not model_filepath:
            return None, None, None # Return None for all outputs if download fails

        model_filename = os.path.basename(model_filepath)

        # Load vae using the existing VAELoader
        return self.vae_loader.load_vae(model_filename)

class OnDemandCLIPLoader:

    @classmethod
    def INPUT_TYPES(s):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("clip_models", []) ]

        return {"required": { 
                                "clip_name": (models,),
                                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], ),
                              },
                "optional": {
                                "device": (["default", "cpu"], {"advanced": True}),
                                "api_key": ("STRING", {"default": None, "multiline": False}),
                                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1})
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "download_clip"
    CATEGORY = "loaders"
    DESCRIPTION = "Load clip models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    def download_clip(self, clip_name, type="stable_diffusion", device="default", api_key=None, download_chunks=None):
        self.clip_loader = CLIPLoader()

        destination_dir = os.path.join(folder_paths.models_dir, "text_encoders")

        model_url = _get_model_url_from_config(clip_name, "clip_models")
        if not model_url:
            return None

        api_key = _get_api_key_for_url(model_url, api_key)

        model_filepath = _download_model(model_url, clip_name, destination_dir, api_key, download_chunks)
        if not model_filepath:
            return None

        model_filename = os.path.basename(model_filepath)

        # Load the checkpoint using the existing CheckpointLoaderSimple
        return self.clip_loader.load_clip(model_filename, type, device)
    


class OnDemandGGUFLoader:

    @classmethod
    def INPUT_TYPES(s):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("gguf_models", []) ]

        return {"required": { 
                                "unet_name": (models,)                        
                              },
                "optional": {
                                "api_key": ("STRING", {"default": None, "multiline": False}),
                                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load gguf models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    def download_unet(self, unet_name, api_key=None, download_chunks=None):
        if module_gguf is None:
            logger.error(f"UnetLoaderGGUF class not available. Ensure ComfyUI-GGUF is installed correctly.")
            return None
        
        self.gguf_loader = module_gguf.nodes.UnetLoaderGGUF()

        destination_dir = os.path.join(folder_paths.models_dir, "unet")

        model_url = _get_model_url_from_config(unet_name, "gguf_models")
        if not model_url:
            return None

        api_key = _get_api_key_for_url(model_url, api_key)

        model_filepath = _download_model(model_url, unet_name, destination_dir, api_key, download_chunks)
        if not model_filepath:
            return None

        model_filename = os.path.basename(model_filepath)

        # Load the gguf using the existing UnetLoaderGGUF
        return self.gguf_loader.load_unet(model_filename)
