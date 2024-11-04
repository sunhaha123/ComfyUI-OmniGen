# ComfyUI-OmniGen

A ComfyUI custom node implementation of [OmniGen](https://github.com/VectorSpaceLab/OmniGen), a powerful text-to-image generation and editing model.

<details>
<summary><h2>Featrues</h2></summary>
  
- Text-to-Image Generation
- Image Editing
- Support for Multiple Input Images
- Memory Optimization Options
- Flexible Image Size Control
</details>

## Installation
  1. open the terminal on the ComfyUI `ComfyUI/custom_nodes` folder
  2. Run `git clone https://github.com/1038lab/ComfyUI-OmniGen.git`
  3. restart ComfyUI

## Install the required Python packages:
  1. open the terminal on the `ComfyUI/custom_nodes/AILab_OmniGen` folder
  2. `..\..\..\python_embeded\python.exe -m pip install -r requirements.txt`

<details>
<summary><h3>Auto-Download Feature</h3></summary>

The node includes automatic downloading of:
1. OmniGen code from GitHub repository
2. Model weights from Hugging Face

No manual file downloading is required. The node will handle everything automatically on first use.</details>
>[!IMPORTANT]
>The first time you use this custom node, it will automatically download the model from Hugging Face. Please be patient, as the download size is approximately 15.5 GB, and it may take some time to complete.
>
>Alternatively, you can manually download the model from Hugging Face at the following link:
>Download OmniGen-v1 from [Hugging Face](https://huggingface.co/Shitao/OmniGen-v1/tree/main)
>After downloading, place the model in the following directory: `comfyui/models/LLM/OmniGen-v1`

### Exsample workflows
Simple useage for text to image & image to image
![Simple useage for text to image & image to image](/Examples/omnigen_1.png)

Generate 2 image and combine image
![Simple useage for text to image & image to image](/Examples/omnigen_2a.png)


<details>
<summary><h2>Example prompts:</h2></summary>
  
| Prompt | Image_1 | Image_2 | Image_3 | Output |
| ------ | ------ | ------ | ------ | ------ |
| 20yo woman looking at viewer |  |  |  | <img src="/Examples/imgs/wm1.png" width="125"> |
| Transform `image_1` into an oil painting | <img src="/Examples/imgs/wm1.png" width="100"> |  |  | <img src="/Examples/imgs/wm1op.png" width="125"> |
| Transform `image_2` into an Anime | <img src="/Examples/imgs/m1.png" width="100"> |  |  | <img src="/Examples/imgs/m1a.png" width="125"> |
| the girl in `image_1` sitting on rock on top of the mountain. | <img src="/Examples/imgs/wm1.png" width="100"> |  |  | <img src="/Examples/imgs/wm1mt.png" width="125"> |
| A woman from `image_1` and a man from `image_2` are sitting across from each other at a cozy coffee shop, each holding a cup of coffee and engaging in conversation. | <img src="/Examples/imgs/wm1.png" width="100"> | <img src="/Examples/imgs/m1.png" width="100"> |  | <img src="/Examples/imgs/cs.png" width="300"> |
| Combine `image1` and `image2` in anime style. | <img src="/Examples/imgs/wm1.png" width="100"> | <img src="/Examples/imgs/m1.png" width="100"> |  | <img src="/Examples/imgs/anime.png" width="300"> |
</details>
<details>
<summary><h2>Using Images in Prompts and Settings</h2></summary>
  
You can reference input images in your prompt using either format:
- `<img><|image_1|>`,`</img><img><|image_2|></img>`,`<img><|image_3|></img>`
- `image_1`, `image_2`, `image_3`
- `image1`, `image2`, `image3`

## Usage
The node will automatically download required files on first use:
- OmniGen code from GitHub
- Model weights from Hugging Face (Shitao/OmniGen-v1)
  
### Input Parameters
- `prompt`: Text description of the desired image
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: Text guidance scale (default: 2.5)
- `img_guidance_scale`: Image guidance scale (default: 1.6)
- `max_input_image_size`: Maximum size for input images (default: 1024)
- `width/height`: Output image dimensions (default: 1024x1024)
- `seed`: Random seed for reproducibility

### Memory Optimization Options
- `separate_cfg_infer`: Separate inference process for different guidance (default: True)
- `offload_model`: Offload model to CPU to reduce memory usage (default: True)
- `use_input_image_size_as_output`: Match output size to input image (default: False) 
</details>
<details>
<summary><h2>Creadits</h2></summary>

- Original OmniGen Model: [VectorSpaceLab/OmniGen](https://github.com/VectorSpaceLab/OmniGen)
- Model Weights: [Shitao/OmniGen-v1](https://huggingface.co/Shitao/OmniGen-v1)
</deatils>
