import gradio as gr
from PIL import Image
import numpy as np
import cv2
import os
import datetime
from imgcraft import Editor

# ===================================================================
# CẤU HÌNH VÀ KHỞI TẠO
# ===================================================================
GALLERY_PATH = "/content/drive/MyDrive/ImgCraft_Gallery"
print("Initializing Gradio App...")
comfyui_editor = Editor()
print("ComfyUI Editor is ready.")
os.makedirs(GALLERY_PATH, exist_ok=True)


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
    img_edited = cv2.cvtColor(np.array(img_edited_pil), cv2.COLOR_RGB2BGR)
    img_original = cv2.cvtColor(np.array(img_original_pil), cv2.COLOR_RGB2BGR)
    gray_edited = cv2.cvtColor(img_edited, cv2.COLOR_BGR2GRAY)
    gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    try:
        detector = cv2.AKAZE_create()
        kpts1, descs1 = detector.detectAndCompute(gray_edited, None)
        kpts2, descs2 = detector.detectAndCompute(gray_original, None)
        if descs1 is None or descs2 is None or len(descs1) < 4 or len(descs2) < 4: raise ValueError("Not enough keypoints.")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descs1, descs2), key=lambda x: x.distance)
        if len(matches) < 4: raise ValueError("Not enough matches.")
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H_coarse, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_coarse is None: raise ValueError("findHomography failed.")
    except Exception as e:
        print(f"Coarse alignment failed: {e}. Returning unaligned image.")
        return img_edited_pil
    h_orig, w_orig = img_original.shape[:2]
    aligned_cv = cv2.warpPerspective(img_edited, H_coarse, (w_orig, h_orig))
    return Image.fromarray(cv2.cvtColor(aligned_cv, cv2.COLOR_BGR2RGB))

# ===================================================================
# HÀM QUẢN LÝ THƯ VIỆN
# ===================================================================
def get_gallery_data():
    pairs_dict = {}
    if not os.path.exists(GALLERY_PATH): return []
    for filename in os.listdir(GALLERY_PATH):
        try:
            if filename.endswith("_original.png"):
                timestamp = filename.replace("_original.png", "")
                if timestamp not in pairs_dict: pairs_dict[timestamp] = {}
                pairs_dict[timestamp]['original'] = os.path.join(GALLERY_PATH, filename)
            elif filename.endswith("_aligned.png"):
                timestamp = filename.replace("_aligned.png", "")
                if timestamp not in pairs_dict: pairs_dict[timestamp] = {}
                pairs_dict[timestamp]['aligned'] = os.path.join(GALLERY_PATH, filename)
        except Exception: continue
    gallery_previews = [
        (pairs_dict[ts]['original'], f"ID: {ts}")
        for ts in sorted(pairs_dict.keys(), reverse=True)
        if 'original' in pairs_dict[ts] and 'aligned' in pairs_dict[ts]
    ]
    return gallery_previews

def refresh_gallery():
    return gr.update(value=get_gallery_data())

def delete_pair_by_id(timestamp_to_delete):
    if not timestamp_to_delete:
        gr.Warning("Không có ID nào được chọn để xóa.")
        return
    try:
        original_path = os.path.join(GALLERY_PATH, f"{timestamp_to_delete}_original.png")
        aligned_path = os.path.join(GALLERY_PATH, f"{timestamp_to_delete}_aligned.png")
        if os.path.exists(original_path): os.remove(original_path)
        if os.path.exists(aligned_path): os.remove(aligned_path)
        gr.Info(f"Đã xóa cặp ảnh ID: {timestamp_to_delete}")
    except Exception as e:
        gr.Warning(f"Lỗi khi xóa ảnh: {e}")

# ===================================================================
# HÀM CHÍNH CHO GRADIO
# ===================================================================
def process_and_align_and_save(image_np, target_width, target_height, progress=gr.Progress()):
    progress(0, desc="Bước 1/4: Đang thay đổi kích thước ảnh gốc...")
    original_pil = Image.fromarray(image_np)
    resized_original_pil = resize_image(original_pil, target_width, target_height)
    progress(0.2, desc="Bước 2/4: Đang xử lý ảnh qua ComfyUI...")
    processed_pil = comfyui_editor.process(resized_original_pil)
    progress(0.8, desc="Bước 3/4: Đang khớp ảnh kết quả...")
    aligned_pil = align_images(processed_pil, resized_original_pil)
    progress(0.9, desc="Bước 4/4: Đang lưu vào Google Drive...")
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_save_path = os.path.join(GALLERY_PATH, f"{timestamp}_original.png")
        aligned_save_path = os.path.join(GALLERY_PATH, f"{timestamp}_aligned.png")
        resized_original_pil.save(original_save_path)
        aligned_pil.save(aligned_save_path)
        gr.Info("Cặp ảnh đã được lưu thành công vào Google Drive!")
    except Exception as e:
        gr.Warning(f"Lỗi khi lưu vào Google Drive: {e}")
    progress(1.0, desc="Hoàn thành!")
    return resized_original_pil, processed_pil, aligned_pil, aligned_pil

# ===================================================================
# XÂY DỰNG GIAO DIỆN GRADIO CHÍNH
# ===================================================================
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 90% !important;}") as demo:
    gr.Markdown("# 🎨 Quy trình Xử lý & Khớp ảnh Tự động (với Google Drive)")
    with gr.Tabs():
        with gr.TabItem("⚙️ Xử lý Ảnh"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="1. Tải lên Ảnh Gốc")
                    with gr.Accordion("Tùy chọn Kích thước", open=True):
                        target_width = gr.Number(label="Chiều rộng Tối đa (pixel)", value=896)
                        target_height = gr.Number(label="Chiều cao Tối đa (pixel)", value=1344)
                    run_button = gr.Button("🚀 Bắt đầu Xử lý & Khớp ảnh", variant="primary")
                with gr.Column(scale=3):
                    gr.Markdown("### Kết quả Xử lý Gần nhất")
                    with gr.Tabs():
                        with gr.TabItem("So sánh Trước & Sau"):
                            with gr.Row():
                                output_original_resized = gr.Image(label="Ảnh Gốc (đã resize)", interactive=False)
                                output_aligned = gr.Image(label="Ảnh Cuối cùng (đã khớp)", interactive=False)
                        with gr.TabItem("Các bước Trung gian"):
                            with gr.Row():
                                output_processed = gr.Image(label="Ảnh sau khi qua ComfyUI (chưa khớp)", interactive=False)
                                output_aligned_2 = gr.Image(label="Ảnh Cuối cùng (đã khớp)", interactive=False)

        with gr.TabItem("🖼️ Thư viện Ảnh (Gallery)"):
            gr.Markdown("Nhấn vào một ảnh trong thư viện bên trái để xem chi tiết và quản lý ở bên phải.")
            with gr.Row():
                # CỘT THƯ VIỆN (1/3)
                with gr.Column(scale=1):
                    refresh_button = gr.Button("🔄 Tải lại Thư viện")
                    # Lưới 3 cột
                    gallery_view = gr.Gallery(label="Ảnh gốc đã xử lý", columns=3, height="auto")
                
                # CỘT CHI TIẾT (2/3) - LUÔN HIỂN THỊ
                with gr.Column(scale=2):
                    gr.Markdown("### Chi tiết Cặp ảnh")
                    # Vùng hiển thị khi chưa chọn gì
                    with gr.Column(visible=True) as placeholder_view:
                        gr.Markdown("*<center>⬅️ Vui lòng chọn một ảnh từ thư viện bên trái để xem chi tiết.</center>*")
                    
                    # Vùng hiển thị khi đã chọn ảnh
                    with gr.Column(visible=False) as detail_view:
                        selected_id = gr.Textbox(label="ID đang xem", interactive=False)
                        with gr.Row():
                            detail_original = gr.Image(label="Ảnh Gốc (đã resize)")
                            detail_aligned = gr.Image(label="Ảnh Đã Khớp")
                        delete_button = gr.Button("🗑️ Xóa cặp ảnh này", variant="stop")

    # === ĐỊNH NGHĨA CÁC SỰ KIỆN ===
    
    # HÀM PHỤ ĐỂ LẤY CHI TIẾT KHI CHỌN ẢNH TRONG GALLERY
    def get_details(evt: gr.SelectData):
        caption = evt.value['caption']
        timestamp = caption.replace("ID: ", "")
        original_path = os.path.join(GALLERY_PATH, f"{timestamp}_original.png")
        aligned_path = os.path.join(GALLERY_PATH, f"{timestamp}_aligned.png")
        return {
            placeholder_view: gr.update(visible=False), # Ẩn placeholder
            detail_view: gr.update(visible=True),      # Hiện vùng chi tiết
            selected_id: timestamp,
            detail_original: original_path,
            detail_aligned: aligned_path
        }
    
    # HÀM PHỤ ĐỂ XÓA VÀ CẬP NHẬT GIAO DIỆN
    def delete_and_refresh(timestamp_to_delete):
        delete_pair_by_id(timestamp_to_delete)
        return {
            gallery_view: refresh_gallery(),
            placeholder_view: gr.update(visible=True),  # Hiện lại placeholder
            detail_view: gr.update(visible=False)       # Ẩn vùng chi tiết
        }

    # SỰ KIỆN CHÍNH
    run_button.click(
        fn=process_and_align_and_save,
        inputs=[input_image, target_width, target_height],
        outputs=[output_original_resized, output_processed, output_aligned, output_aligned_2]
    ).then(fn=refresh_gallery, inputs=None, outputs=gallery_view)
    
    gallery_view.select(
        fn=get_details,
        inputs=None,
        outputs=[placeholder_view, detail_view, selected_id, detail_original, detail_aligned]
    )

    delete_button.click(
        fn=delete_and_refresh,
        inputs=[selected_id],
        outputs=[gallery_view, placeholder_view, detail_view]
    )
    
    refresh_button.click(fn=refresh_gallery, inputs=None, outputs=gallery_view)
    
    demo.load(fn=refresh_gallery, inputs=None, outputs=gallery_view)

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
