import gradio as gr
from PIL import Image
import numpy as np
import cv2
import math
from imgcraft import Editor

# ===================================================================
# KHỞI TẠO CÁC THÀNH PHẦN CHÍNH (Tải mô hình một lần)
# ===================================================================
print("Initializing Gradio App...")
try:
    comfyui_editor = Editor()
    print("ComfyUI Editor is ready.")
except Exception as e:
    print(f"Fatal error during Editor initialization: {e}")
    comfyui_editor = None

# ===================================================================
# CÁC HÀM XỬ LÝ ẢNH
# ===================================================================
def resize_image(image_pil, target_width, target_height):
    original_width, original_height = image_pil.size
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    ratio = min(width_ratio, height_ratio)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    return image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

def align_images(img_edited_pil, img_original_pil):
    # Chuyển đổi PIL -> OpenCV BGR
    img_edited = cv2.cvtColor(np.array(img_edited_pil), cv2.COLOR_RGB2BGR)
    img_original = cv2.cvtColor(np.array(img_original_pil), cv2.COLOR_RGB2BGR)

    # Bước 1: Căn chỉnh Thô
    gray_edited = cv2.cvtColor(img_edited, cv2.COLOR_BGR2GRAY)
    gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    try:
        detector = cv2.AKAZE_create()
        kpts1, descs1 = detector.detectAndCompute(gray_edited, None)
        kpts2, descs2 = detector.detectAndCompute(gray_original, None)
        
        if descs1 is None or descs2 is None or len(descs1) < 4 or len(descs2) < 4:
             raise ValueError("Not enough keypoints/descriptors found for coarse alignment.")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descs1, descs2), key=lambda x: x.distance)
        
        if len(matches) < 4:
            raise ValueError("Not enough matches for homography estimation.")

        src_pts = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H_coarse, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_coarse is None:
            raise ValueError("findHomography failed in coarse alignment.")

    except Exception as e:
        print(f"Coarse alignment failed: {e}")
        # Nếu căn thô thất bại, trả về ảnh chưa khớp để người dùng thấy
        return img_edited_pil

    h_orig, w_orig = img_original.shape[:2]
    aligned_cv = cv2.warpPerspective(img_edited, H_coarse, (w_orig, h_orig))

    # Chuyển đổi lại OpenCV BGR -> PIL
    return Image.fromarray(cv2.cvtColor(aligned_cv, cv2.COLOR_BGR2RGB))


# ===================================================================
# HÀM CHÍNH CHO GRADIO
# ===================================================================
def process_and_align(image_np, target_width, target_height, progress=gr.Progress()):
    if comfyui_editor is None:
        raise gr.Error("Editor không được khởi tạo. Vui lòng kiểm tra log Colab.")
    
    # 1. Tải và Resize ảnh gốc
    progress(0, desc="Bước 1/3: Đang thay đổi kích thước ảnh gốc...")
    original_pil = Image.fromarray(image_np)
    resized_original_pil = resize_image(original_pil, target_width, target_height)

    # 2. Xử lý qua ComfyUI
    progress(0.3, desc="Bước 2/3: Đang xử lý ảnh qua ComfyUI (có thể mất vài phút)...")
    processed_pil = comfyui_editor.process(resized_original_pil)

    # 3. Khớp ảnh đã xử lý với ảnh gốc đã resize
    progress(0.8, desc="Bước 3/3: Đang khớp ảnh kết quả...")
    aligned_pil = align_images(processed_pil, resized_original_pil)

    progress(1.0, desc="Hoàn thành!")
    return processed_pil, aligned_pil, aligned_pil # Trả 3 ảnh cho 3 output

# ===================================================================
# XÂY DỰNG GIAO DIỆN GRADIO
# ===================================================================
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 90% !important;}") as demo:
    gr.Markdown("# 🎨 Quy trình Xử lý & Khớp ảnh Tự động")
    gr.Markdown("Tải lên ảnh gốc, chọn kích thước, sau đó chương trình sẽ tự động xử lý qua ComfyUI và khớp kết quả với ảnh gốc đã được resize.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="1. Tải lên Ảnh Gốc")
            with gr.Accordion("Tùy chọn Kích thước", open=True):
                target_width = gr.Number(label="Chiều rộng Tối đa (pixel)", value=896)
                target_height = gr.Number(label="Chiều cao Tối đa (pixel)", value=1344)
            
            run_button = gr.Button("🚀 Bắt đầu Xử lý & Khớp ảnh", variant="primary")

        with gr.Column(scale=3):
            gr.Markdown("### Kết quả")
            with gr.Tabs():
                with gr.TabItem("So sánh Trước & Sau"):
                    # Gradio không có image comparison tốt như Streamlit, ta dùng 2 ảnh riêng
                    gr.Markdown("So sánh ảnh **đã khớp** với ảnh **gốc đã resize**.")
                    with gr.Row():
                        # Cần một ảnh gốc để so sánh, ta sẽ tạo nó trong hàm xử lý
                        # nhưng hiện tại để trống, ta sẽ cập nhật sau.
                        # Đây là một mẹo nhỏ: ta sẽ trả về ảnh đã khớp cho cả 2 output này
                        # để so sánh.
                        output_aligned_for_compare = gr.Image(label="Ảnh Gốc (đã resize)", interactive=False)
                        output_aligned = gr.Image(label="Ảnh Cuối cùng (đã khớp)", interactive=False)
                
                with gr.TabItem("Các bước Trung gian"):
                    with gr.Row():
                        output_processed = gr.Image(label="Ảnh sau khi qua ComfyUI (chưa khớp)", interactive=False)
                        output_aligned_2 = gr.Image(label="Ảnh Cuối cùng (đã khớp)", interactive=False)

    # Logic xử lý
    run_button.click(
        fn=process_and_align,
        inputs=[input_image, target_width, target_height],
        outputs=[output_processed, output_aligned, output_aligned_2]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
