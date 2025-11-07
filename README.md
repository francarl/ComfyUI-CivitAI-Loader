# ComfyUI OnDemand Loaders

This is a suite of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to download and apply various models directly from Civitai or HuggingFace. It simplifies your workflow by managing model downloads automatically, so you don't have to manually download files and place them in the correct folders.

## Features

*   **Automatic Downloading**: Downloads models from Civitai/HuggingFace URLs specified in a configuration file.
*   **On-Demand Loading**: Models are downloaded only when they are needed for a workflow.
*   **Centralized Configuration**: Manage your list of LoRAs, Checkpoints, VAEs, and more from a single `config.json` file.
*   **No Re-downloads**: Checks if a model file already exists before attempting to download it.
*   **Console Progress Bar**: Displays a tqdm progress bar in the console during download.
*   **Private Model Support**: Access private or early-access models using your Civitai or HuggingFace API key. Environment variables (`CIVITAI_TOKEN`, `HUGGINGFACE_TOKEN`) are also supported.
*   **Seamless Integration**: Functions as standard loader nodes within the ComfyUI interface.

## Available Nodes

This package includes the following nodes:
*   `OnDemand Lora Loader`
*   `OnDemand Checkpoint Loader`
*   `OnDemand VAE Loader`
*   `OnDemand UNET Loader`
*   `OnDemand CLIP Loader`
*   `OnDemand GGUF Loader` (Note: Requires [ComfyUI-GGUF](https://github.com/jquesnelle/ComfyUI-GGUF) to be installed.)

## Installation with ComfyUI-Manager
Search for "OnDemand-Loaders" in custom node list.

## Manual installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/francarl/ComfyUI-OnDemand-Loaders.git
    ```
3.  Install the required dependencies:
    ```bash
    cd ComfyUI-OnDemand-Loaders
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

## Usage

### 1. Configure Your Models

After installation, create and configure a `config.json` file to list the models you want to access.

1.  In the `ComfyUI/custom_nodes/ComfyUI-OnDemand-Loaders/` directory, create a file named `config.json`. You can use the `example/config.json` file as a template. If `config.json` does not exist, the nodes will fall back to a default example configuration.
2.  Add your models to `config.json`, organized by type (`loras`, `checkpoints`, `vae_models`, etc.). Each entry requires a `name` (which will appear in the node's dropdown menu) and a `url` (the Civitai/HuggingFace **download** link).

You can get the download link from a model's page on Civitai/HuggingFace by right-clicking the download button and copying the link address.

**Example `config.json`:**
```json
{ 
    "loras": [
        {
            "name": "epi_noiseoffset",
            "url": "https://civitai.com/api/download/models/16576"
        },
        {
            "name": "Studio Ghibli Style LoRA",
            "url": "https://civitai.com/api/download/models/7657"
        }
    ],
    "checkpoints": [
        {
            "name": "v1-5-pruned-emaonly-fp16",
            "url": "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors"
        }
    ],
    "vae_models": [
        {
            "name": "SDXL VAE",
            "url": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
        }
    ],
    "clip_models": [
        {
            "name": "OpenCLIP ViT-H-14",
            "url": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin"
        }
    ],
    "gguf_models": [
        {
            "name": "Wan2.2-T2V-A14B-HighNoise-Q2_K",
            "url": "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf"
        }
    ]
}
```

### 2. Add the Nodes in ComfyUI

1.  In ComfyUI, double-click on the canvas to open the search menu.
2.  Search for and add one of the **`OnDemand ... Loader`** nodes (e.g., `OnDemand Checkpoint Loader`).
3.  Select the desired model from the dropdown. The first time you queue a workflow with a new model, it will be automatically downloaded to the corresponding directory inside `ComfyUI/models/`.
4.  Connect the node in your workflow just as you would with a standard loader. An example workflow can be found in `example/workflow_example.json`.

After modifying `config.json`, press "R" to reload configuration and update workflow without restarting ComfyUI.

### 3. Common Node Options

*   `..._name`: A dropdown menu to select the model you configured in `config.json`.
*   `api_key` (optional): Your API key for Civitai or HuggingFace. Use this to download private or early-access models. Alternatively, you can set the `CIVITAI_TOKEN` or `HUGGINGFACE_TOKEN` environment variables.
*   `download_chunks` (optional): The chunk size (in KB) for downloading files. The default is 4KB.

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.
