import sys
import os.path as osp
import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
import requests
import folder_paths
import tempfile
import shutil

# Define all path constants
class Paths:
    ROOT_DIR = osp.dirname(__file__)
    MODELS_DIR = folder_paths.models_dir
    LLM_DIR = osp.join(MODELS_DIR, "LLM")
    OMNIGEN_DIR = osp.join(LLM_DIR, "OmniGen-v1")
    OMNIGEN_CODE_DIR = osp.join(ROOT_DIR, "OmniGen")
    TMP_DIR = osp.join(ROOT_DIR, "tmp")
    MODEL_FILE = osp.join(OMNIGEN_DIR, "model.safetensors")

# Ensure necessary directories exist
os.makedirs(Paths.LLM_DIR, exist_ok=True)
sys.path.append(Paths.ROOT_DIR)

class ailab_OmniGen:
    def __init__(self):
        self._ensure_code_exists()
        self._ensure_model_exists()
        try:
            from OmniGen import OmniGenPipeline
            self.OmniGenPipeline = OmniGenPipeline
        except ImportError as e:
            print(f"Error importing OmniGen: {e}")
            raise RuntimeError("Failed to import OmniGen. Please check if the code was downloaded correctly.")
        
    def _ensure_code_exists(self):
        """Ensure OmniGen code exists, download from Hugging Face if not"""
        try:
            if not osp.exists(Paths.OMNIGEN_CODE_DIR):
                print("Downloading OmniGen code from Hugging Face...")
                
                # Files to download from Hugging Face
                files = [
                    "model.py",
                    "pipeline.py",
                    "processor.py",
                    "scheduler.py",
                    "transformer.py",
                    "utils.py",
                    "__init__.py"
                ]
                
                os.makedirs(Paths.OMNIGEN_CODE_DIR, exist_ok=True)
                base_url = "https://huggingface.co/spaces/Shitao/OmniGen/raw/main/OmniGen/"
                
                for file in files:
                    url = base_url + file
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(osp.join(Paths.OMNIGEN_CODE_DIR, file), 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {file}")
                    else:
                        raise RuntimeError(f"Failed to download {file}: {response.status_code}")
                
                print("OmniGen code setup completed")
                
                if Paths.OMNIGEN_CODE_DIR not in sys.path:
                    sys.path.append(Paths.OMNIGEN_CODE_DIR)
                        
            else:
                print("OmniGen code already exists")
                    
        except Exception as e:
            print(f"Error downloading OmniGen code: {e}")
            raise RuntimeError(f"Failed to download OmniGen code: {str(e)}")

    def _ensure_model_exists(self):
        """Ensure model file exists, download if not"""
        try:
            if not osp.exists(Paths.MODEL_FILE):
                print("OmniGen model not found, starting download from Hugging Face...")
                os.makedirs(Paths.OMNIGEN_DIR, exist_ok=True)
                snapshot_download(
                    repo_id="Shitao/OmniGen-v1",
                    local_dir=Paths.OMNIGEN_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    token=None,  # Add your token if needed
                    tqdm_class=None,  # This will use default progress bar
                )
                print("OmniGen model downloaded successfully")
            else:
                print("OmniGen model found locally")
        except Exception as e:
            print(f"Error during model initialization: {e}")
            raise RuntimeError(f"Failed to initialize OmniGen model: {str(e)}")

    def _setup_temp_dir(self):
        """Set up temporary directory"""
        if osp.exists(Paths.TMP_DIR):
            shutil.rmtree(Paths.TMP_DIR)
        os.makedirs(Paths.TMP_DIR, exist_ok=True)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if osp.exists(Paths.TMP_DIR):
            shutil.rmtree(Paths.TMP_DIR)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "forceInput": False, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 5.0, "step": 0.1, "round": 0.01}),
                "img_guidance_scale": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 2.0, "step": 0.1, "round": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "separate_cfg_infer": ("BOOLEAN", {"default": True}),
                "offload_model": ("BOOLEAN", {"default": False}),
                "use_input_image_size_as_output": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_input_image_size": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 16}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generation"
    CATEGORY = "🧪AILab/OmniGen"

    def save_input_img(self, image):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=Paths.TMP_DIR) as f:
                img_np = image.numpy()[0] * 255
                img_pil = Image.fromarray(img_np.astype(np.uint8))
                img_pil.save(f.name)
                return f.name
        except Exception as e:
            print(f"Error saving input image: {e}")
            return None

    def _process_prompt_and_images(self, prompt, images):
        """Process prompt and images, return updated prompt and image paths"""
        input_images = []
        
        # Auto-generate prompt if empty but images provided
        if not prompt and any(images):
            prompt = " ".join(f"<img><|image_{i+1}|></img>" for i, img in enumerate(images) if img is not None)
        
        # Process each image
        for i, img in enumerate(images, 1):
            if img is not None:
                img_path = self.save_input_img(img)
                if img_path:
                    input_images.append(img_path)
                    img_tag = f"<img><|image_{i}|></img>"
                    # Handle both image_1 and image1 formats
                    if f"image_{i}" in prompt:
                        prompt = prompt.replace(f"image_{i}", img_tag)
                    elif f"image{i}" in prompt:  # Added support for image1 format
                        prompt = prompt.replace(f"image{i}", img_tag)
                    elif img_tag not in prompt:
                        prompt += f" {img_tag}"
        
        return prompt, input_images

    def generation(self, prompt, num_inference_steps, guidance_scale,
            img_guidance_scale, max_input_image_size, separate_cfg_infer, offload_model,
            use_input_image_size_as_output, width, height, seed, image_1=None, image_2=None, image_3=None):
        try:
            self._setup_temp_dir()
            pipe = self.OmniGenPipeline.from_pretrained(Paths.OMNIGEN_DIR)
            
            # Switch to eager mode only if SDPA is not supported
            if not self._check_sdpa_support():
                if hasattr(pipe, 'text_encoder'):
                    pipe.text_encoder.config.attn_implementation = "eager"
                if hasattr(pipe, 'unet'):
                    pipe.unet.config.attn_implementation = "eager"
            
            # Process prompt and images
            prompt, input_images = self._process_prompt_and_images(prompt, [image_1, image_2, image_3])
            input_images = input_images if input_images else None
            
            print(f"Processing with prompt: {prompt}")
            output = pipe(
                prompt=prompt,
                input_images=input_images,
                guidance_scale=guidance_scale,
                img_guidance_scale=img_guidance_scale,
                num_inference_steps=num_inference_steps,
                separate_cfg_infer=separate_cfg_infer, 
                use_kv_cache=True,
                offload_kv_cache=True,
                offload_model=offload_model,
                use_input_image_size_as_output=use_input_image_size_as_output,
                width=width,
                height=height,
                seed=seed,
                max_input_image_size=max_input_image_size,
            )
            
            img = np.array(output[0]) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)
            return (img,)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise e
        finally:
            self._cleanup_temp_dir()
            torch.cuda.empty_cache()

    def _check_sdpa_support(self):
        """Check if system supports SDPA"""
        try:
            import torch
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return True
        except Exception as e:
            print(f"Warning: SDPA not supported, falling back to eager attention implementation")
        
        return False


NODE_CLASS_MAPPINGS = {
    "ailab_OmniGen": ailab_OmniGen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ailab_OmniGen": "OmniGen 🖼️"
}
