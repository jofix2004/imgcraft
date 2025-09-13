# /imgcraft/core.py
import os
import sys
import torch
import numpy as np
from PIL import Image
import gc
import datetime
import random
from IPython.display import display, Image as IPImage

# --- PHẦN 1: THIẾT LẬP MÔI TRƯỜỜNG VÀ IMPORT ---
comfyui_path = '/content/ComfyUI'
if comfyui_path not in sys.path:
    sys.path.insert(0, comfyui_path)

from nodes import (
    DualCLIPLoader, CLIPTextEncode, VAEEncode, VAEDecode, VAELoader,
    KSamplerAdvanced, ConditioningZeroOut, LoraLoaderModelOnly, LoadImage,
    SaveImage
)
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_sd3 import EmptySD3LatentImage

# --- PHẦN 2: KHỞI TẠO CÁC NODE (NHƯ BIẾN TOÀN CỤC) ---
print("Initializing ComfyUI nodes for imgcraft...")
try:
    clip_loader = DualCLIPLoader()
    unet_loader = UnetLoaderGGUF()
    vae_loader = VAELoader()
    vae_encode = VAEEncode()
    vae_decode = VAEDecode()
    ksampler = KSamplerAdvanced()
    load_lora = LoraLoaderModelOnly()
    load_image_node = LoadImage()
    save_image_node = SaveImage()
    positive_prompt_encode = CLIPTextEncode()
    negative_prompt_encode = ConditioningZeroOut()
    empty_latent_image = EmptySD3LatentImage()
    flux_guidance = FluxGuidance()
    reference_latent = ReferenceLatent()
    print("✅ All nodes initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing nodes: {e}")

# --- PHẦN 3: LỚP EDITOR SỬ DỤNG CÁC NODE ĐÃ KHỞI TẠO ---
class Editor:
    def __init__(self, model_dir="/content/ComfyUI/models"):
        self.model_dir = model_dir # Vẫn giữ lại để tham khảo nếu cần, nhưng không dùng để xây dựng đường dẫn nữa
        self.output_path = None

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _save_image(self, image_tensor, prefix="output"):
        os.makedirs("/content/output", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        output_path = os.path.join("/content/output", filename)
        frame = (image_tensor.cpu().numpy().squeeze() * 255).astype(np.uint8)
        Image.fromarray(frame).save(output_path)
        return output_path

    def _resize_image(self, image_path, target_width=1360, target_height=2048):
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        processed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        processed_filename = f"processed_{os.path.basename(image_path)}"
        processed_path = os.path.join('/content/ComfyUI/input', processed_filename)
        processed_img.save(processed_path)
        return processed_path, new_width, new_height

    def process(self, image_path: str):
        if not os.path.exists(image_path):
            print(f"Lỗi: Không tìm thấy tệp ảnh tại '{image_path}'")
            return

        with torch.inference_mode():
            try:
                resized_path, width, height = self._resize_image(image_path)
                print(f"Image processed to: {width}x{height}")

                # SỬA LỖI: Chỉ truyền tên tệp, không truyền đường dẫn đầy đủ.
                clip = clip_loader.load_clip(
                    "t5xxl_fp8_e4m3fn.safetensors",
                    "clip_l.safetensors",
                    "flux"
                )[0]
                positive_prompt = "Manga cleaning, remove text, remove sfx"
                prompt_encode = positive_prompt_encode.encode(clip, positive_prompt)[0]
                negative = negative_prompt_encode.zero_out(prompt_encode)[0]
                del clip
                self._clear_memory()

                image_tensor = load_image_node.load_image(resized_path)[0]

                # SỬA LỖI: Chỉ truyền tên tệp.
                vae = vae_loader.load_vae("ae.sft")[0]

                latent = vae_encode.encode(vae, image_tensor)[0]
                conditioning = reference_latent.append(prompt_encode, latent)[0]
                positive = flux_guidance.append(conditioning, 2.5)[0]

                # SỬA LỖI: Chỉ truyền tên tệp.
                model = unet_loader.load_unet("flux1-kontext-dev-Q6_K.gguf")[0]

                # SỬA LỖI: Chỉ truyền tên tệp.
                model = load_lora.load_lora_model_only(
                    model, "flux_1_turbo_alpha.safetensors", 1.0
                )[0]
                model = load_lora.load_lora_model_only(
                    model, "AniGa-CleMove-000005.safetensors", 0.8
                )[0]

                output_latent = empty_latent_image.generate(width, height, 1)[0]
                seed = random.randint(0, 2**32 - 1)
                print(f"Editing image with seed: {seed}...")
                image_out_latent = ksampler.sample(
                    model=model, add_noise="enable", noise_seed=seed, steps=8, cfg=1.0,
                    sampler_name="euler", scheduler="simple", positive=positive, negative=negative,
                    latent_image=output_latent, start_at_step=0, end_at_step=1000, return_with_leftover_noise="disable"
                )[0]
                del model
                self._clear_memory()

                decoded_image = vae_decode.decode(vae, image_out_latent)[0]
                del vae
                self._clear_memory()

                self.output_path = self._save_image(decoded_image)
                print(f"\n✅ Processing complete! Image saved to: {self.output_path}")
                display(IPImage(filename=self.output_path))
            except Exception as e:
                print(f"An error occurred during processing: {e}")
            finally:
                self._clear_memory()
