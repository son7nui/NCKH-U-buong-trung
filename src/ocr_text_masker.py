"""
Module tiền xử lý OCR-Guided Inpainting:
Khử chữ chú thích viết tắt ('CY', 'LOV', 'ROV', 'UT', số đo) và thước đo siêu âm (caliper)
Tuyệt đối KHÔNG xóa nhầm các cấu trúc tăng âm bệnh học thực sự (vôi hóa, mỡ, tóc trong Teratoma
hoặc vách bao xơ của u nang buồng trứng).
"""

import cv2
import numpy as np
from pathlib import Path

def mask_text_and_calipers_ocr_guided(img_bgr, ocr_reader=None):
    """
    Che phủ và khử chú thích văn bản và thước đo siêu âm:
    1. Nhận diện caliper màu vàng trên ảnh 2D thông qua không gian màu HSV và lọc diện tích.
    2. Nhận diện bounding box của chữ viết bằng EasyOCR, áp dụng hình thái học giãn nở nhẹ (dilation 2-3px)
       và inpaint bằng giải thuật Telea.
    3. Bảo toàn các vùng tăng âm bệnh học (không inpaint vùng có kích thước lớn hoặc không phải chữ).

    Trả về:
    - img_cleaned: Ảnh sau khi inpainting.
    - combined_mask: Mặt nạ nhị phân các vùng đã inpaint (255 = inpainted, 0 = giữ nguyên).
    """
    img_cleaned = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Khử caliper màu vàng trên ảnh siêu âm 2D
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 100, 100])
    upper_yellow = np.array([38, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Lọc thành phần liên thông: Chỉ nhận diện các nét mảnh nhỏ dạng dấu '+' hoặc đường gạch chấm
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Thước đo siêu âm thường có diện tích rất nhỏ từ 5 đến 250 pixel
        if 5 <= area <= 250:
            combined_mask[labels == i] = 255

    # 2. Nhận diện chữ máy gõ bằng OCR (nếu có reader)
    if ocr_reader is not None:
        try:
            results = ocr_reader.readtext(img_bgr)
            for (bbox, text, prob) in results:
                # Lọc chữ cái viết hoa hoặc số chú thích ngắn của máy siêu âm
                clean_text = text.strip()
                if prob > 0.25 and len(clean_text) > 0:
                    pts = np.array(bbox, dtype=np.int32)
                    bx, by, bw, bh = cv2.boundingRect(pts)
                    # Tiêu chí an toàn: Chiều cao chữ thường < 45 pixel và diện tích vừa phải
                    if bh < 50 and (bw * bh) < (h * w * 0.05):
                        cv2.fillPoly(combined_mask, [pts], 255)
        except Exception as e:
            pass

    # 3. Thực hiện Inpaint bằng giải thuật Telea nếu có vùng cần khử
    if np.sum(combined_mask) > 0:
        # Giãn nhẹ mask 2px để bao trùm viền chữ
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        img_cleaned = cv2.inpaint(img_cleaned, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return img_cleaned, combined_mask

def create_inpainting_qa_panel(sample_img_paths, output_path, ocr_reader=None):
    """
    Tạo panel trực quan kiểm tra chất lượng (QA Panel) so sánh Before vs After Inpainting.
    """
    rows = []
    for p in sample_img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        cleaned, mask = mask_text_and_calipers_ocr_guided(img, ocr_reader)
        
        # Tạo ảnh overlay mask màu đỏ lên ảnh gốc để dễ đối chiếu
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 0] = 0 # B
        mask_color[:, :, 1] = 0 # G
        mask_overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)
        
        combined_row = np.hstack([cv2.resize(img, (256, 256)), 
                                  cv2.resize(mask_overlay, (256, 256)), 
                                  cv2.resize(cleaned, (256, 256))])
        rows.append(combined_row)

    if rows:
        final_panel = np.vstack(rows)
        cv2.imwrite(str(output_path), final_panel)
        return True
    return False
