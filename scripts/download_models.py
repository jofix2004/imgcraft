# /scripts/download_models.py
import os
import subprocess
from pathlib import Path

# Thư mục gốc cho các mô hình trong ComfyUI
COMFYUI_MODEL_DIR = os.getenv("COMFYUI_MODEL_DIR", "/content/ComfyUI/models")

MODELS = {
    "unet": {
        "flux1-kontext-dev-Q6_K.gguf": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/flux1-kontext-dev-Q6_K.gguf",
    },
    "vae": {
        "ae.sft": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/ae.sft",
    },
    "clip": {
        "clip_l.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/clip_l.safetensors",
        "t5xxl_fp8_e4m3fn.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/t5xxl_fp8_e4m3fn.safetensors",
    },
    "loras": {
        "flux_1_turbo_alpha.safetensors": "https://huggingface.co/Isi99999/Upscalers/resolve/main/Flux/flux_1_turbo_alpha.safetensors",
        "AniGa-CleMove-000005.safetensors": "https://huggingface.co/TranLinh2004/AniGa-CleMove/resolve/main/AniGa-CleMove-000005.safetensors",
    }
}

def download_model(url: str, dest_dir: str, filename: str = None) -> bool:
    """Tải file với aria2c và hiển thị tiến trình."""
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]

        cmd = [
            'aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M',
            '-d', dest_dir, '-o', filename, url
        ]
        
        print(f"Downloading {filename}...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✓ Downloaded {filename} successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading {filename}: {e.stderr.decode().strip()}")
        return False
    except FileNotFoundError:
        print("✗ aria2c not found. Please install it using 'sudo apt-get install aria2c'")
        return False

def download_all_models():
    """Tải tất cả các mô hình cần thiết."""
    print("--- Starting Model Download ---")
    all_successful = True
    for model_type, files in MODELS.items():
        dest_folder = os.path.join(COMFYUI_MODEL_DIR, model_type)
        for filename, url in files.items():
            if not download_model(url, dest_folder, filename):
                all_successful = False
    
    if all_successful:
        print("\n✅ All models downloaded successfully!")
    else:
        print("\n⚠️ Some models failed to download.")
        
if __name__ == "__main__":
    download_all_models()
