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
# Đường dẫn đến thư mục lưu trữ trên Google Drive
GALLERY_PATH = "/content/drive/MyDrive/ImgCraft_Gallery"

print("Initializing Gradio App...")
# Tạo một instance Editor "rỗng". Các mô hình sẽ được tải trong hàm process.
comfyui_editor = Editor()
print("ComfyUI Editor is ready.")
# Đảm bảo thư mục gallery tồn tại
os.makedirs(GALLERY_PATH, exist_ok=True)


# ===================================================================
# CÁC HÀM XỬ LÝ ẢNH (Không thay đổi)
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
        if descs1 is None or descs2 is None or len(descs1) < 4 or len(descs2) < 4:
             raise ValueError("Not enough keypoints for coarse alignment.")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descs1, descs2), key=lambda x: x.distance)
        if len(matches) < 4:
            raise ValueError("Not enough matches for homography.")
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H_coarse, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_coarse is None:
            raise ValueError("findHomography failed.")
    except Exception as e:
        print(f"Coarse alignment failed: {e}. Returning unaligned image.")
        return img_edited_pil

    h_orig, w_orig = img_original.shape[:2]
    aligned_cv = cv2.warpPerspective(img_edited, H_coarse, (w_orig, h_orig))
    return Image.fromarray(cv2.cvtColor(aligned_cv, cv2.COLOR_BGR2RGB))

# ===================================================================
# HÀM QUẢN LÝ THƯ VIỆN
# ===================================================================
def get_gallery_pairs():
    """Quét thư mục Drive và trả về danh sách các cặp ảnh."""
    pairs = {}
    if not os.path.exists(GALLERY_PATH):
        return []
    
    for filename in os.listdir(GALLERY_PATH):
        if filename.endswith((".png", ".jpg")):
            parts = filename.split('_')
            if len(parts) == 2:
                timestamp, ftype = parts[0], parts[1].split('.')[0]
                if timestamp not in pairs:
                    pairs[timestamp] = {}
                pairs[timestamp][ftype] = os.path.join(GALLERY_PATH, filename)

    # Lọc ra những cặp đầy đủ và sắp xếp theo thứ tự mới nhất
    full_pairs = [
        (pairs[ts]['original'], pairs[ts]['aligned'])
        for ts in sorted(pairs.keys(), reverse=True)
        if 'original' in pairs[ts] and 'aligned' in pairs[ts]
    ]
    return full_pairs

def delete_pair(original_path):
    """Xóa một cặp ảnh dựa trên đường dẫn của ảnh gốc."""
    try:
        aligned_path = original_path.replace("_original.png", "_aligned.png")
        if os.path.exists(original_path):
            os.remove(original_path)
        if os.path.exists(aligned_path):
            os.remove(aligned_path)
        gr.Info(f"Đã xóa cặp ảnh!")
    except Exception as e:
        gr.Warning(f"Lỗi khi xóa ảnh: {e}")
    return refresh_gallery() # Tải lại thư viện sau khi xóa

# ===================================================================
# HÀM CHÍNH CHO GRADIO
# ===================================================================
def process_and_align_and_save(image_np, target_width, target_height, progress=gr.Progress()):
    # 1. Resize ảnh gốc
    progress(0, desc="Bước 1/4: Đang thay đổi kích thước ảnh gốc...")
    original_pil = Image.fromarray(image_np)
    resized_original_pil = resize_image(original_pil, target_width, target_height)

    # 2. Xử lý qua ComfyUI
    progress(0.2, desc="Bước 2/4: Đang xử lý ảnh qua ComfyUI...")
    processed_pil = comfyui_editor.process(resized_original_pil)

    # 3. Khớp ảnh
    progress(0.8, desc="Bước 3/4: Đang khớp ảnh kết quả...")
    aligned_pil = align_images(processed_pil, resized_original_pil)

    # 4. Lưu vào Google Drive
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
# HÀM XÂY DỰNG GIAO DIỆN THƯ VIỆN ĐỘNG
# ===================================================================
def refresh_gallery():
    """Tạo lại giao diện thư viện từ dữ liệu mới nhất trên Drive."""
    pairs = get_gallery_pairs()
    # Trả về một component gr.Column chứa các cặp ảnh
    # hoặc một thông báo nếu thư viện trống
    if not pairs:
        return gr.Column(visible=True, value=[gr.Markdown("*Thư viện đang trống. Hãy xử lý một ảnh để bắt đầu!*")])

    # Tạo một danh sách các component để hiển thị
    gallery_items = []
    for orig_path, aligned_path in pairs:
        with gr.Blocks() as item_block: # Dùng Blocks để nhóm các component
            with gr.Row(variant="panel"):
                gr.Image(orig_path, label="Gốc (đã resize)", height=256)
                gr.Image(aligned_path, label="Đã khớp", height=256)
                with gr.Column(min_width=100):
                    filename = os.path.basename(orig_path).replace("_original.png", "")
                    gr.Markdown(f"**ID:**\n`{filename}`")
                    delete_button = gr.Button("🗑️ Xóa", variant="stop")
        
        # Liên kết sự kiện click của nút xóa với hàm delete_pair
        # Dùng lambda để truyền đường dẫn vào hàm
        delete_button.click(
            fn=lambda p=orig_path: delete_pair(p),
            inputs=None,
            outputs=gallery_container # Nút xóa sẽ kích hoạt việc làm mới toàn bộ thư viện
        )
        gallery_items.append(item_block)
        
    return gr.Column(visible=True, value=gallery_items)


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
            gr.Markdown("Xem lại các cặp ảnh đã xử lý được lưu trên Google Drive của bạn.")
            refresh_button = gr.Button("🔄 Tải lại Thư viện")
            gallery_container = gr.Column() # Vùng chứa động cho thư viện
    
    # === ĐỊNH NGHĨA CÁC SỰ KIỆN ===
    
    # 1. Khi nhấn nút xử lý
    run_button.click(
        fn=process_and_align_and_save,
        inputs=[input_image, target_width, target_height],
        outputs=[output_original_resized, output_processed, output_aligned, output_aligned_2]
    ).then( # Sau khi xử lý xong, tự động làm mới thư viện
        fn=refresh_gallery,
        inputs=None,
        outputs=gallery_container
    )
    
    # 2. Khi nhấn nút tải lại thư viện
    refresh_button.click(
        fn=refresh_gallery,
        inputs=None,
        outputs=gallery_container
    )
    
    # 3. Khi ứng dụng tải lần đầu tiên, tự động tải thư viện
    demo.load(
        fn=refresh_gallery,
        inputs=None,
        outputs=gallery_container
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
