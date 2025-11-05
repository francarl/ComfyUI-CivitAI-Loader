from .nodes import OnDemandLoraLoader, OnDemandUNETLoader, OnDemandCheckpointLoader

NODE_CLASS_MAPPINGS = {
    "OnDemandLoraLoader": OnDemandLoraLoader,
    "OnDemandUNETLoader": OnDemandUNETLoader,
    "OnDemandCheckpointLoader": OnDemandCheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnDemandLoraLoader": "OnDemand Lora Loader",
    "OnDemandUNETLoader": "OnDemand UNET Loader",
    "OnDemandCheckpointLoader": "OnDemand Checkpoint Loader"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


# Module metadata
__version__ = "1.0.0"
__author__ = "francarl"
__description__ = "A suite of nodes for on-demand loading of Models, VAE, Clip and Loras in ComfyUI."
