# /imgcraft/core.py
import os
import sys
import torch
import numpy as np
from PIL import Image
import gc
import datetime
import random

# --- PHẦN 1: THIẾT LẬP MÔI TRƯỜNG VÀ IMPORT ---
comfyui_path = '/content/ComfyUI'
if comfyui_path not in sys.path:
    sys.path.insert(0, comfyui_path)

from nodes import (
    DualCLIPLoader, CLIPTextEncode, VAEEncode, VAEDecode, VAELoader,
    KSamplerAdvanced, ConditioningZeroOut, LoraLoaderModelOnly, LoadImage
)
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_sd3 import EmptySD3LatentImage

# --- PHẦN 2: KHỞI TẠO CÁC NODE (NHƯ BIẾN TOÀN CỤC) ---
print("Initializing ComfyUI nodes for imgcraft...")
try:
    clip_loader_node = DualCLIPLoader()
    unet_loader_node = UnetLoaderGGUF()
    vae_loader_node = VAELoader()
    vae_encode_node = VAEEncode()
    vae_decode_node = VAEDecode()
    ksampler_node = KSamplerAdvanced()
    load_lora_node = LoraLoaderModelOnly()
    load_image_node = LoadImage()
    positive_prompt_encode_node = CLIPTextEncode()
    negative_prompt_encode_node = ConditioningZeroOut()
    empty_latent_image_node = EmptySD3LatentImage()
    flux_guidance_node = FluxGuidance()
    reference_latent_node = ReferenceLatent()
    print("✅ All nodes initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing nodes: {e}")

# --- PHẦN 3: LỚP EDITOR PHIÊN BẢN ỔN ĐỊNH ---
class Editor:
    def __init__(self):
        # KHÔNG tải trước bất kỳ mô hình nào. Lớp này giờ đây "vô tri".
        pass

    def _clear_memory(self):
        """Hàm dọn dẹp bộ nhớ tích cực."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _pil_to_tensor(self, image: Image.Image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def _tensor_to_pil(self, tensor: torch.Tensor):
        return Image.fromarray((tensor.cpu().numpy().squeeze() * 255).astype(np.uint8))

    def process(self, image_pil: Image.Image):
        width, height = image_pil.size
        print(f"Processing image with resolution: {width}x{height}")

        with torch.inference_mode():
            try:
                # ======================================================
                # MÔ HÌNH ĐƯỢC TẢI LẠI TỪ ĐẦU Ở MỖI LẦN CHẠY
                # ======================================================
                print("\n--- Loading models for this run (Stable Mode) ---")

                print("Loading CLIP...")
                clip = clip_loader_node.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
                
                positive_prompt = "Manga cleaning, remove text, remove sfx"
                prompt_encode = positive_prompt_encode_node.encode(clip, positive_prompt)[0]
                negative = negative_prompt_encode_node.zero_out(prompt_encode)[0]
                del clip # << GIẢI PHÓNG NGAY LẬP TỨC
                
                image_tensor = self._pil_to_tensor(image_pil)

                print("Loading VAE...")
                vae = vae_loader_node.load_vae("ae.sft")[0]
                
                latent = vae_encode_node.encode(vae, image_tensor)[0]
                conditioning = reference_latent_node.append(prompt_encode, latent)[0]
                positive = flux_guidance_node.append(conditioning, 2.5)[0]

                print("Loading UNet and applying LoRAs...")
                model = unet_loader_node.load_unet("flux1-kontext-dev-Q6_K.gguf")[0]
                model = load_lora_node.load_lora_model_only(model, "flux_1_turbo_alpha.safetensors", 1.0)[0]
                model = load_lora_node.load_lora_model_only(model, "AniGa-CleMove-000005.safetensors", 0.8)[0]

                output_latent = empty_latent_image_node.generate(width, height, 1)[0]
                seed = random.randint(0, 2**32 - 1)
                
                print(f"Starting rendering with seed: {seed}...")
                image_out_latent = ksampler_node.sample(
                    model=model, add_noise="enable", noise_seed=seed, steps=8, cfg=1.0,
                    sampler_name="euler", scheduler="simple", positive=positive, negative=negative,
                    latent_image=output_latent, start_at_step=0, end_at_step=1000, return_with_leftover_noise="disable"
                )[0]
                del model # << GIẢI PHÓNG NGAY LẬP TỨC
                
                print("Decoding latents...")
                decoded_tensor = vae_decode_node.decode(vae, image_out_latent)[0]
                del vae # << GIẢI PHÓNG NGAY LẬP TỨC
                
                return self._tensor_to_pil(decoded_tensor)
            
            except Exception as e:
                print(f"An error occurred during ComfyUI processing: {e}")
                raise
            finally:
                # Luôn dọn dẹp triệt để sau khi kết thúc
                print("Cleaning up all temporary memory...")
                self._clear_memory()
