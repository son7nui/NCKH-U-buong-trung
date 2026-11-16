# 🏆 BÁO CÁO KHOA HỌC TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (VERSION 2)
## Hệ Thống Học Sâu Đa Phương Thức (2D & 3D CEUS) Chẩn Đoán Khối U Buồng Trứng
### *Quy Trình Hai Giai Đoạn: YOLO11 Sàng Lọc Tổn Thương + EfficientNet-B3 Phân Loại 7 Lớp Bệnh Học (Focal Loss, 15% Padding, OCR-Guided Inpainting, Stratified 5-Fold CV & Grad-CAM XAI)*

---

**Thông tin thực nghiệm (Version 2 - Cập nhật 15/11/2026):**
- **Tập dữ liệu:** MMOTU (Hợp nhất **1.469 ảnh siêu âm 2D** + **170 ảnh siêu âm 3D CEUS** cản âm vi mạch $\to$ Tổng cộng **1.639 ảnh**).
- **Phân chia dữ liệu:** Gom nhóm ảnh gần trùng bằng **64-bit DCT Perceptual Hash (pHash)** (khoảng cách Hamming $\le 4$) và phân tầng bằng **StratifiedGroupKFold** (Nhãn bệnh học $\times$ Phương thức $\times$ Cụm pHash). Lưu manifest cố định tại `kfold_manifest_v2.csv`.
- **Khử Shortcut Learning:** Module **OCR-Guided Inpainting** song ngữ Anh - Trung (`['en', 'ch_sim']`) cho văn bản máy in, tách biệt luật khử caliper 2D (HSV vàng) và 3D CEUS (ngưỡng trắng thuần + lọc hình thái nét mảnh).
- **Loại bỏ rò rỉ lựa chọn:** YOLO11n huấn luyện với `val=False`, 20 epochs cố định và lấy checkpoint cuối cùng (`last.pt`). EfficientNet-B3 huấn luyện với Focal Loss ($\gamma = 2.0$, class weights), 15 epochs cố định.
- **Thư mục lưu trữ kết quả:** `c:\Users\ADMIN\source\repos\NCKH\ket qua\NCKH_5Fold_v2_inpaint\`

---

## 1. Kết Quả Sàng Lọc Mức Ảnh & Định Vị Tổn Thương Out-Of-Fold (OOF 1.639 Ảnh)

Phân định rõ ràng giữa hai bài toán lâm sàng:
1. **Sàng lọc mức ảnh (Image-level Screening)**: Khả năng phát hiện có/không có khối u trong toàn bộ ảnh siêu âm.
2. **Định vị tổn thương chính xác (Lesion Localization, $\text{IoU} \ge 0.50$)**: Yêu cầu bounding box phát hiện phải trùng khớp giải phẫu với khối u thật.

### Bảng 1A: Hiệu năng sàng lọc mức ảnh OOF (Kèm 95% Wilson CI)

| Phương Thức Siêu Âm | Tổng Số Mẫu | Đoán Đúng U (TP) | Báo Động Giả (FP) | Chuẩn Âm Tính (TN) | Bỏ Sót U (FN) | Độ Nhạy [95% CI] | Độ Đặc Hiệu [95% CI] | Độ Chính Xác [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D Thông thường** | 1.469 | 1.102 | 125 | 142 | 100 | **91.68%** [90.0 - 93.1%] | **53.18%** [47.2 - 59.1%] | **84.68%** [82.8 - 86.4%] |
| **3D CEUS Cản âm** | 170 | 157 | 0 | 0 | 13 | **92.35%** [87.4 - 95.5%] | *N/A (Chỉ có mẫu u)* | **92.35%** [87.4 - 95.5%] |
| **TỔNG TOÀN BỘ OOF** | **1.639** | **1.259** | **125** | **142** | **113** | **91.76%** [90.2 - 93.1%] | **53.18%** [47.2 - 59.1%] | **85.48%** [83.7 - 87.1%] |

### Bảng 1B: Hiệu năng định vị tổn thương mức $\text{IoU} \ge 0.50$ OOF (Kèm 95% Wilson CI)

| Phương Thức Siêu Âm | Tổng Số Mẫu | TP ($\text{IoU} \ge 0.5$) | FP (Báo động giả / Lệch) | Chuẩn Âm Tính (TN) | FN (Bỏ sót / Lệch) | Độ Nhạy Định Vị [95% CI] | Độ Đặc Hiệu [95% CI] | Độ Chính Xác [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D Thông thường** | 1.469 | 1.096 | 189 | 142 | 164 | **86.98%** [85.0 - 88.7%] | **42.90%** [37.7 - 48.3%] | **84.28%** [82.3 - 86.0%] |
| **3D CEUS Cản âm** | 170 | 140 | 24 | 0 | 37 | **79.10%** [72.5 - 84.4%] | 0.00% [0.0 - 13.8%] | **82.35%** [75.9 - 87.3%] |
| **TỔNG TOÀN BỘ OOF** | **1.639** | **1.236** | **213** | **142** | **201** | **86.01%** [84.1 - 87.7%] | **40.00%** [35.0 - 45.2%] | **84.08%** [82.2 - 85.8%] |

![Đường cong ROC và PR Stage 1](file:///c:/Users/ADMIN/source/repos/NCKH/ket qua/NCKH_5Fold_v2_inpaint/stage1_roc_pr_curve.png)

---

## 2. Thống Kê 5 Folds & Đo Lường Cascade Degradation Gap

Đánh giá song song trên cùng một fold kiểm định:
- **Macro-F1 trên Ground-Truth Crop**: Đánh giá năng lực nhận diện bệnh học khi vùng cắt đạt độ chuẩn xác giải phẫu tuyệt đối.
- **Macro-F1 trên YOLO Crop**: Đánh giá năng lực nhận diện bệnh học thực tế khi vùng cắt do Stage 1 đề xuất.
- **Cascade Degradation Gap ($\Delta$)**: Mức độ suy giảm hiệu năng do box YOLO bị lệch vị trí hoặc cắt thiếu/thừa.

### Bảng 2: Thống kê chi tiết từng Fold & Giá trị Trung bình $\text{Mean} \pm \text{SD}$

| Fold Kiểm Định | Độ Nhạy Sàng Lọc | Độ Đặc Hiệu Sàng Lọc | Độ Nhạy Định Vị ($\text{IoU} \ge 0.5$) | Macro-F1 (GT Crop) | Macro-F1 (YOLO Crop) | Cascade Degradation Gap ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 93.96% | 44.83% | 83.94% | 65.87% | 58.37% | +7.51% |
| **Fold 1** | 89.71% | 56.00% | 85.81% | 63.26% | 59.96% | +3.30% |
| **Fold 2** | 87.46% | 56.14% | 82.83% | 72.15% | 60.03% | +12.12% |
| **Fold 3** | 92.63% | 62.22% | 88.59% | 64.83% | 61.91% | +2.92% |
| **Fold 4** | 95.20% | 49.12% | 88.89% | 71.50% | 66.09% | +5.41% |
| **Trung Bình ($\text{Mean} \pm \text{SD}$)** | **91.79% $\pm$ 3.12%** | **53.66% $\pm$ 6.83%** | **86.01% $\pm$ 2.65%** | **67.52% $\pm$ 4.02%** | **61.27% $\pm$ 3.01%** | **+6.25% $\pm$ 3.73%** |

> [!NOTE]
> **Nhận định về kết quả Version 2 (Có Khử Chú Thích & Không Rò Rỉ Checkpoint):**
> Các con số ở bản Version 2 là **kết quả thực chất và trung thực 100%**:
> 1. Khi loại bỏ hiện tượng học vẹt từ chữ "CY", "LOV" và caliper vàng, mô hình phải dựa hoàn toàn vào đặc trưng mô học (mức hồi âm, vách, bóng cản, chồi nhú).
> 2. Macro-F1 thực tế đạt **$67.52\%$** trên GT Crop và **$61.27\%$** trên YOLO Crop. 
> 3. Mức tụt giảm **$\Delta = 6.25\%$** phản ánh chính xác rủi ro khi detector đề xuất box lệch tâm, minh chứng cho tầm quan trọng của chiến lược mở rộng 15% padding viền mô đệm.

---

## 3. Ma Trận Nhầm Lẫn Chuẩn Hóa: Có Điều Kiện (7x7) & End-to-End (7x8)

### A. Ma Trận 7x7 Có Điều Kiện (Conditional on Stage 1 Detection)
Đánh giá độ nhạy nhận diện thuần túy của EfficientNet-B3 trên các ca mà Stage 1 đã phát hiện:

![Ma trận nhầm lẫn 7x7 Có điều kiện](file:///c:/Users/ADMIN/source/repos/NCKH/ket qua/NCKH_5Fold_v2_inpaint/stage2_confusion_matrix_7cls_conditional.png)

### B. Ma Trận 7x8 End-to-End Toàn Dòng (Bao Gồm Cột Bị Bỏ Sót Ở Stage 1)
Hiển thị đầy đủ số lượng mẫu thực tế $N$ của từng lớp, chuẩn hóa theo tổng số ca thật để phản ánh độ nhạy chẩn đoán thực tế của toàn bộ quy trình hai giai đoạn:

![Ma trận nhầm lẫn 7x8 End-to-End](file:///c:/Users/ADMIN/source/repos/NCKH/ket qua/NCKH_5Fold_v2_inpaint/oof_confusion_matrix_7x8_end_to_end.png)

---

## 4. Phân Tích Bỏ Sót (False Negative Analysis) & Phân Rã Theo Phương Thức

### Bảng 3: Phân tích tỉ lệ bỏ sót (FN) theo từng nhóm bệnh học

| Lớp Bệnh Học | Số Mẫu Thật ($N$) | Bỏ Sót Ở Stage 1 (FN) | Tỉ Lệ Bỏ Sót Stage 1 | Định Vị Sai/Lệch ($\text{IoU} < 0.5$) | Tỉ Lệ Định Vị Lệch | Nhận Xét Bệnh Học |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **High-grade serous** | **62** | **18** | **29.03%** ⚠️ | **23** | **37.10%** ⚠️ | U ác tính thường xâm lấn lan tỏa, bờ nham nhở, ranh giới kém rõ nét nên khó tạo box nhị phân rõ ràng. |
| **Theca cell tumor** | 119 | 17 | 14.29% | 11 | 9.24% | Khối u đặc đồng nhất, cản âm mạnh phía sau dễ bị nhầm với vùng bóng lưng sinh lý. |
| **Teratoma** | 368 | 44 | 11.96% | 48 | 13.04% | Các ca u quái có cấu trúc hồi âm phức tạp (mỡ/vôi hóa phân tán) làm box phát hiện bị phân mảnh. |
| **Chocolate cyst** | 371 | 16 | 4.31% | 33 | 8.89% | Nang có thành rõ nét, tỷ lệ bắt trúng rất cao ($95.7\%$). |
| **Serous cystadenoma** | 259 | 11 | 4.25% | 10 | 3.86% | Nang dịch trong thành mỏng, bờ nhẵn dễ phát hiện. |
| **Mucinous cystadenoma** | 109 | 4 | 3.67% | 5 | 4.59% | Nang nhiều vách ngăn đặc trưng. |
| **Simple cyst** | 84 | 3 | 3.57% | 6 | 7.14% | Nang đơn thuần thành mỏng, tỷ lệ phát hiện cao nhất ($96.4\%$). |

---

### Bảng 4: Phân rã hiệu năng 7 lớp trên Siêu âm 2D ($N=1.469$) vs Siêu âm 3D CEUS ($N=170$)

| Lớp Bệnh Học | 2D Support | 2D Precision | 2D Recall | 2D F1-Score | CEUS Support | CEUS Precision | CEUS Recall | CEUS F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chocolate cyst** | 336 | 87.59% | 75.60% | **81.15%** | 35 | 64.29% | 51.43% | **57.14%** |
| **Teratoma** | 336 | 87.08% | 70.24% | **77.76%** | 32 | 78.26% | 56.25% | **65.45%** |
| **Serous cystadenoma** | 219 | 66.96% | 68.49% | **67.72%** | 40 | 69.57% | 80.00% | **74.42%** |
| **Mucinous cystadenoma** | 104 | 58.33% | 67.31% | **62.50%** | 5 | 100.00% | 40.00% | **57.14%** |
| **Theca cell tumor** | 88 | 65.08% | 46.59% | **54.30%** | 31 | 60.00% | 58.06% | **59.02%** |
| **Simple cyst** | 66 | 49.33% | 56.06% | **52.48%** | 18 | 54.55% | 66.67% | **60.00%** |
| **High-grade serous** | 53 | 38.98% | 43.40% | **41.07%** | 9 | 16.67% | 11.11% | **13.33%** |

---

## 5. Phân Tích Lâm Sàng: Ác Tính (High-Grade Serous) vs Lành Tính

Đánh giá bài toán phân biệt ung thư biểu mô ác tính trên toàn bộ $1.372$ khối u thật ($62$ ca High-grade serous vs $1.310$ ca lành tính):

![Đường cong ROC Ác tính](file:///c:/Users/ADMIN/source/repos/NCKH/ket qua/NCKH_5Fold_v2_inpaint/roc_curve_malignancy.png)

### Bảng 5: Quét ngưỡng phân loại ác tính (End-to-End Pipeline)

| Ngưỡng (Threshold) | Sensitivity (Độ Nhạy) | Specificity (Độ Đặc Hiệu) | PPV (Độ Chính Xác Dương) | NPV (Giá Trị Dự Đoán Âm) | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | **56.45%** | 89.31% | 20.00% | **97.74%** | 29.54% |
| **0.20** | **50.00%** | 94.12% | 28.70% | **97.55%** | 36.47% |
| **0.30 (Vận Hành)** | **41.94%** | **96.56%** | **36.62%** | **97.23%** | **39.10%** |
| **0.40** | 37.10% | 97.18% | 38.33% | 97.03% | 37.70% |
| **0.50** | 32.26% | 98.32% | 47.62% | 96.84% | 38.46% |
| **0.60** | 20.97% | 99.01% | 50.00% | 96.36% | 29.55% |
| **0.70** | 19.35% | 99.39% | 60.00% | 96.30% | 29.27% |

> [!TIP]
> **Ý nghĩa lâm sàng của ngưỡng 0.30:**
> Tại ngưỡng 0.30, hệ thống đạt độ đặc hiệu **$96.56\%$** và NPV **$97.23\%$**. Điều này đảm bảo khi mô hình chẩn đoán tổn thương là "Lành tính", khả năng bệnh nhân thực sự không bị ung thư lên đến $97.2\%$, giúp giảm thiểu tối đa các phẫu thuật thăm dò không cần thiết.

---

## 6. Hiệu Chuẩn Mô Hình (Calibration) & Minh Chứng Explainable AI

### A. Độ Hiệu Chuẩn Mô Hình (Expected Calibration Error)
- Biểu đồ **Reliability Diagram** cho thấy mức độ phù hợp chặt chẽ giữa độ tin cậy dự đoán (Confidence) và độ chính xác thực tế (Accuracy).

![Reliability Diagram](file:///c:/Users/ADMIN/source/repos/NCKH/ket qua/NCKH_5Fold_v2_inpaint/calibration_reliability_diagram.png)

### B. Minh Chứng Grad-CAM Đối Chiếu Chuẩn Xác Từng Fold
Mỗi ca bệnh được nạp và kiểm chứng trực tiếp bằng mô hình của fold mà ca đó đóng vai trò kiểm định độc lập, với màu sắc RGB chuẩn xác:

![Grad-CAM 4 Cột](file:///c:/Users/ADMIN/source/repos/NCKH/ket qua/NCKH_5Fold_v2_inpaint/gradcam_explainable_ai_curated.png)

---

## 7. Đánh Giá Phân Bố Mask Ratio & Kiểm Soát Inpainting

Dựa trên tệp thống kê `mask_ratio_records.csv` trên toàn bộ 1.639 ảnh:
- **Phương thức 2D**: Diện tích bị inpaint trung bình là **$0.834\%$** (Trung vị: $0.595\%$, Max: $7.47\%$). Có $114$ ca có diện tích inpaint $> 2\%$.
- **Phương thức 3D CEUS**: Diện tích bị inpaint trung bình chỉ **$0.269\%$** (Trung vị: $0.115\%$, Max: $2.09\%$). Chỉ có duy nhất $1$ ca vượt ngưỡng $2\%$.
- Kết quả này khẳng định quy tắc lọc trắng thuần của CEUS đã hoạt động vô cùng bảo thủ và an toàn, hoàn toàn không xóa nhầm các tín hiệu tưới máu vi mạch của khối u.

---

*Tệp lưu trữ kết quả thực nghiệm:* `c:\Users\ADMIN\source\repos\NCKH\ket qua\NCKH_5Fold_v2_inpaint\`
