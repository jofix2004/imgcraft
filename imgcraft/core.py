# @title 1.5. Ghi đè file core.py với phiên bản đã sửa lỗi
# @markdown Chạy cell này để thay thế hoàn toàn file core.py gốc.
# /imgcraft/core.py
import os
import sys

# --- KEY FIX ---
# Thay đổi thư mục làm việc hiện tại sang thư mục gốc của ComfyUI.
# Điều này cực kỳ quan trọng vì nhiều import nội bộ của ComfyUI
# giả định rằng chương trình đang được chạy từ đây.
comfyui_path = '/content/ComfyUI'

# os.chdir(comfyui_path) # <<< ĐÂY LÀ DÒNG GÂY LỖI.
# Vô hiệu hóa để tránh làm thay đổi thư mục làm việc của toàn bộ chương trình,
# gây ra lỗi ModuleNotFoundError.

# Đảm bảo đường dẫn này cũng có trong sys.path để chắc chắn.
# Đây là cách chính xác và an toàn để Python tìm thấy các module của ComfyUI.
if comfyui_path not in sys.path:
    sys.path.insert(0, comfyui_path)
# --- END OF FIX ---


# Bây giờ, các import khác có thể tiếp tục một cách an toàn
import torch
import numpy as np
from PIL import Image
import gc
import datetime
import random
from IPython.display import display, Image as IPImage

# Import các node cần thiết từ ComfyUI (BÂY GIỜ SẼ HOẠT ĐỘNG)
from nodes import (
    DualCLIPLoader, CLIPTextEncode, VAEEncode, VAEDecode, VAELoader,
    KSamplerAdvanced, ConditioningZeroOut, LoraLoaderModelOnly, LoadImage,
    SaveImage
)
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_images import ImageScale

class Editor:
    def __init__(self, model_dir="/content/ComfyUI/models"):
        """Khởi tạo trình chỉnh sửa và các node cần thiết."""
        self.model_dir = model_dir
        self._initialize_nodes()
        self.output_path = None

    def _initialize_nodes(self):
        """Khởi tạo các instance của node ComfyUI."""
        self.clip_loader = DualCLIPLoader()
        self.unet_loader = UnetLoaderGGUF()
        self.vae_loader = VAELoader()
        self.vae_encode = VAEEncode()
        self.vae_decode = VAEDecode()
        self.ksampler = KSamplerAdvanced()
        self.load_lora = LoraLoaderModelOnly()
        self.load_image = LoadImage()
        self.save_image_node = SaveImage()
        self.positive_prompt_encode = CLIPTextEncode()
        self.negative_prompt_encode = ConditioningZeroOut()
        self.empty_latent_image = EmptySD3LatentImage()
        self.flux_guidance = FluxGuidance()
        self.reference_latent = ReferenceLatent()
        self.image_scaler = ImageScale()

    def _clear_memory(self):
        """Giải phóng bộ nhớ GPU và RAM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _save_image(self, image_tensor, prefix="output"):
        """Lưu tensor ảnh thành file và trả về đường dẫn."""
        os.makedirs("/content/output", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        output_path = os.path.join("/content/output", filename)

        frame = (image_tensor.cpu().numpy().squeeze() * 255).astype(np.uint8)
        Image.fromarray(frame).save(output_path)
        return output_path

    def _resize_image(self, image_path, target_width=1360, target_height=2048):
        """Thay đổi kích thước ảnh để vừa với khung mục tiêu mà không làm méo."""
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        processed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Lưu vào thư mục input của ComfyUI
        processed_filename = f"processed_{os.path.basename(image_path)}"
        processed_path = os.path.join('/content/ComfyUI/input', processed_filename)
        processed_img.save(processed_path)

        return processed_path, new_width, new_height

    def process(self, image_path: str):
        """
        Thực thi quy trình chỉnh sửa ảnh với các tham số cố định.

        Args:
            image_path (str): Đường dẫn đến ảnh cần xử lý.
        """
        if not os.path.exists(image_path):
            print(f"Lỗi: Không tìm thấy tệp ảnh tại '{image_path}'")
            return

        with torch.inference_mode():
            try:
                # 1. Tải và xử lý kích thước ảnh
                print("Resizing image...")
                resized_path, width, height = self._resize_image(image_path)
                print(f"Image processed to: {width}x{height}")

                # 2. Tải mô hình và encode prompt
                print("Loading Text Encoder...")
                clip = self.clip_loader.load_clip(
                    os.path.join(self.model_dir, "clip/t5xxl_fp8_e4m3fn.safetensors"),
                    os.path.join(self.model_dir, "clip/clip_l.safetensors"),
                    "flux"
                )[0]

                positive_prompt = "Manga cleaning, remove text, remove sfx"
                prompt_encode = self.positive_prompt_encode.encode(clip, positive_prompt)[0]
                negative = self.negative_prompt_encode.zero_out(prompt_encode)[0]
                del clip
                self._clear_memory()

                # 3. Tải VAE và encode ảnh thành latent
                print("Loading VAE and encoding image...")
                image_tensor = self.load_image.load_image(resized_path)[0]

                vae = self.vae_loader.load_vae(os.path.join(self.model_dir, "vae/ae.sft"))[0]
                latent = self.vae_encode.encode(vae, image_tensor)[0]

                conditioning = self.reference_latent.append(prompt_encode, latent)[0]
                positive = self.flux_guidance.append(conditioning, 2.5)[0]

                # 4. Tải Unet và LoRA
                print("Loading Unet Model and LoRAs...")
                model = self.unet_loader.load_unet(
                    os.path.join(self.model_dir, "unet/flux1-kontext-dev-Q6_K.gguf")
                )[0]

                # Áp dụng Turbo LoRA
                model = self.load_lora.load_lora_model_only(
                    model,
                    os.path.join(self.model_dir, "loras/flux_1_turbo_alpha.safetensors"),
                    1.0  # strength
                )[0]

                # Áp dụng LoRA tùy chỉnh
                model = self.load_lora.load_lora_model_only(
                    model,
                    os.path.join(self.model_dir, "loras/AniGa-CleMove-000005.safetensors"),
                    0.8  # strength
                )[0]

                # 5. Tạo latent rỗng cho output
                output_latent = self.empty_latent_image.generate(width, height, 1)[0]

                # 6. Sampling
                seed = random.randint(0, 2**32 - 1)
                print(f"Editing image with seed: {seed}...")
                image_out_latent = self.ksampler.sample(
                    model=model,
                    add_noise="enable",
                    noise_seed=seed,
                    steps=8, # Sử dụng 8 bước vì có Turbo LoRA
                    cfg=1.0,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=positive,
                    negative=negative,
                    latent_image=output_latent,
                    start_at_step=0,
                    end_at_step=1000,
                    return_with_leftover_noise="disable"
                )[0]
                del model
                self._clear_memory()

                # 7. Decode và lưu ảnh
                print("Decoding latents...")
                decoded_image = self.vae_decode.decode(vae, image_out_latent)[0]
                del vae
                self._clear_memory()

                self.output_path = self._save_image(decoded_image)
                print(f"\n✅ Processing complete! Image saved to: {self.output_path}")
                display(IPImage(filename=self.output_path))

            except Exception as e:
                print(f"An error occurred during processing: {e}")
            finally:
                self._clear_memory()

# Ghi file thành công
print("✅ File /content/imgcraft/imgcraft/core.py đã được cập nhật với phiên bản đã sửa lỗi.")
