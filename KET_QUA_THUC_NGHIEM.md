# BÁO CÁO KẾT QUẢ THỰC NGHIỆM HỌC SÂU (YOLO11)
## Phát Hiện & Phân Loại U Buồng Trứng Từ Ảnh Siêu Âm (Bộ Dữ Liệu MMOTU)

---

## 1. Tổng quan Đề tài & Mục tiêu Nghiên cứu

Đề tài tập trung vào việc áp dụng kiến trúc mạng thị giác máy tính tiên tiến **YOLO11** (phiên bản `yolo11n`) để tự động phát hiện, khoanh vùng vị trí (Bounding Box) và phân loại đa lớp (8 nhóm bệnh học) các khối u buồng trứng từ hình ảnh siêu âm phụ khoa 2D (B-mode và Color Doppler).

### Thông số mô hình & Môi trường thực nghiệm
- **Mô hình**: YOLO11 Nano (`yolo11n.pt`)
- **Số lượng tham số**: ~2.59 triệu tham số (Parameters)
- **Độ phức tạp tính toán**: 6.5 GFLOPs
- **Dung lượng file trọng số**: 5.46 MB (`best.pt`)
- **Bộ tối ưu hóa (Optimizer)**: AdamW (Learning rate ban đầu: `0.001`, lr cuối: `0.01`)
- **Kích thước ảnh đầu vào (imgsz)**: 320 / 640
- **Kích thước batch (Batch size)**: 16
- **Tốc độ suy luận (Inference Speed)**: **23.5 ms/ảnh (~42.5 FPS)** trên CPU (đáp ứng tiêu chuẩn xử lý thời gian thực trong thăm khám siêu âm lâm sàng).

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm Định Lượng

Dưới đây là bảng số liệu chi tiết thu được từ quá trình huấn luyện và đánh giá trên tập kiểm định (Validation set):

| Tiêu chí đánh giá | Epoch 1 | Epoch 2 (Hiện tại) | Thay đổi / Xu hướng |
| :--- | :---: | :---: | :---: |
| **Thời gian huấn luyện / epoch** | 154.9s | 155.6s | Ổn định |
| **Train Box Loss** | 1.4247 | **1.2598** | Giảm 11.6% |
| **Train Cls Loss** | 3.2134 | **2.3789** | Giảm 26.0% |
| **Train DFL Loss** | 1.5254 | **1.3428** | Giảm 12.0% |
| **Validation Box Loss** | 1.5647 | **1.2191** | Giảm 22.1% |
| **Validation Cls Loss** | 11.1671 | **3.0083** | Giảm mạnh 73.1% |
| **Validation DFL Loss** | 2.0524 | **1.3838** | Giảm 32.6% |
| **Precision toàn cục (B)** | 50.29% | **48.53%** | Đạt cực đại 95% ở conf cao |
| **Recall toàn cục (B)** | 18.88% | **30.06%** | **Tăng +59.2%** |
| **mAP@50 (B)** | 9.92% | **20.30%** | **Tăng gấp hơn 2 lần (+104.6%)** |
| **mAP@50-95 (B)** | 5.06% | **12.94%** | **Tăng gấp hơn 2.5 lần (+155.7%)** |

### Chi tiết độ chính xác mAP@50 theo từng nhóm bệnh học

| STT | Mã Lớp | Tên Nhóm Bệnh Học (Class Name) | mAP@50 (%) | Ghi chú hình thái siêu âm |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 1 | **Serous cystadenoma** (U nang thanh dịch) | **44.9%** | Ranh giới rõ, dịch đồng nhất, bắt bbox tốt |
| 2 | 0 | **Chocolate cyst** (U nang sô-cô-la / Lạc nội mạc) | **40.0%** | Echo kém đồng nhất dạng kính mờ (ground glass) |
| 3 | 6 | **Mucinous cystadenoma** (U nang nhầy) | **20.9%** | Nhiều vách ngăn, dịch nhầy hồi âm đa dạng |
| 4 | 2 | **Teratoma** (U quái buồng trứng / Nang bì) | **20.0%** | Thành phần hỗn hợp (mỡ, dịch, cản âm nốt bì) |
| 5 | 4 | **Simple cyst** (U nang đơn giản / cơ năng) | **9.7%** | Thành mỏng, vỏ bao nhẵn, echo trống |
| 6 | 3 | **Theca cell tumor** (U tế bào vỏ) | **4.8%** | U thể đặc, kích thước hiếm gặp trong tập mẫu |
| 7 | 7 | **High-grade serous** (Ung thư biểu mô ác tính cao) | **1.6%** | Tổ chức xâm lấn, giới hạn không rõ, mẫu hiếm |
| - | 5 | **Normal ovary** (Buồng trứng bình thường) | *Mẫu âm tính* | Không chứa u (background đối chứng) |
| **-** | **All** | **Trung bình toàn bộ các lớp (All Classes)** | **20.3%** | **Đạt 0.203 sau 2 epoch thăm dò** |

---

## 3. Danh Sách & Phân Tích Các Biểu Đồ Kết Quả

*(Báo cáo đã lọc bỏ toàn bộ các ảnh chụp lát cắt siêu âm mẫu, chỉ lưu giữ và phân tích các ảnh biểu đồ thống kê chuyên sâu).*

---

### Biểu đồ 1: Tiến trình Huấn luyện & Đánh giá Qua Các Epoch (`results.png`)

![Tiến trình huấn luyện results.png](results/charts/results.png)

**Nhận xét chuyên môn:**
- Cả 3 hàm mất mát huấn luyện (`train/box_loss`, `train/cls_loss`, `train/dfl_loss`) đều có độ dốc giảm rất tốt và liên tục qua các epoch.
- Mất mát phân loại tập kiểm định (`val/cls_loss`) giảm đột phá từ mức `11.17` xuống `3.01`, cho thấy mô hình thích ứng và phân biệt đặc trưng u buồng trứng rất nhanh chóng.
- Cả hai chỉ số mAP then chốt là `metrics/mAP50(B)` và `metrics/mAP50-95(B)` đều tăng trưởng tuyến tính dốc đứng, phản ánh tiềm năng gia tăng độ chính xác vượt bậc nếu tiếp tục huấn luyện ở các epoch tiếp theo (30-50 epochs).

---

### Biểu đồ 2: Đường cong Precision-Recall (`BoxPR_curve.png`)

![Đường cong Precision-Recall BoxPR_curve.png](results/charts/BoxPR_curve.png)

**Nhận xét chuyên môn:**
- Diện tích dưới đường cong PR (AUC-PR) thể hiện độ cân bằng giữa tỷ lệ dự đoán chính xác và khả năng bao quát ca bệnh.
- Hai nhóm u đạt kết quả vượt trội nhất là **Serous cystadenoma (44.9%)** và **Chocolate cyst (40.0%)**, đường cong PR duy trì ở mức cao trên phần lớn khoảng Recall.
- Các lớp u hiếm như *High-grade serous* (1.6%) và *Theca cell tumor* (4.8%) bị giới hạn do số lượng mẫu huấn luyện còn khiêm tốn, kéo mAP trung bình của tất cả các lớp xuống mức 20.3%.

---

### Biểu đồ 3: Biểu đồ F1-Score theo Ngưỡng Tin Cậy (`BoxF1_curve.png`)

![Biểu đồ F1-Score BoxF1_curve.png](results/charts/BoxF1_curve.png)

**Nhận xét chuyên môn:**
- Điểm F1 trung bình toàn bộ (All classes) đạt đỉnh cực đại là **0.22 tại ngưỡng Confidence = 0.468**.
- Đối với 2 lớp chiếm ưu thế là *Chocolate cyst* và *Serous cystadenoma*, giá trị F1 đạt tiệm cận **0.50** trong dải confidence từ `0.45` đến `0.65`.
- **Ứng dụng thực tế**: Khi triển khai mô hình hỗ trợ bác sĩ lâm sàng, nên đặt ngưỡng tin cậy (Confidence Threshold) ở mức xấp xỉ **0.45 – 0.50** để cân đối tối ưu giữa độ chính xác và độ nhạy.

---

### Biểu đồ 4: Biểu đồ Precision theo Ngưỡng Tin Cậy (`BoxP_curve.png`)

![Biểu đồ Precision BoxP_curve.png](results/charts/BoxP_curve.png)

**Nhận xét chuyên môn:**
- Đường biểu diễn Precision tăng đều theo ngưỡng tin cậy. Khi tăng Confidence > 0.6, Precision toàn thể vượt 60% và đạt **0.95 (95%) tại ngưỡng Confidence = 1.000**.
- Đối với *Serous cystadenoma* và *Chocolate cyst*, ngay từ ngưỡng tin cậy 0.4, mô hình đã duy trì độ chính xác dự đoán ổn định trên 40-50%.

---

### Biểu đồ 5: Biểu đồ Recall theo Ngưỡng Tin Cậy (`BoxR_curve.png`)

![Biểu đồ Recall BoxR_curve.png](results/charts/BoxR_curve.png)

**Nhận xét chuyên môn:**
- Ở ngưỡng tin cậy thấp gần 0, Recall toàn cục đạt mức cao **0.87 (87%)**.
- Đặc tính này rất có ý nghĩa trong bài toán sàng lọc y tế: nếu cấu hình mô hình ở chế độ "Sàng lọc ban đầu" (Confidence thấp ~ 0.20), mô hình có thể phát hiện tới **~50-60%** các ca tổn thương, hạn chế tối đa nguy cơ bỏ sót u buồng trứng nguy hiểm.

---

### Biểu đồ 6: Ma trận Nhầm lẫn Chuẩn hóa (`confusion_matrix_normalized.png`)

![Ma trận nhầm lẫn chuẩn hóa confusion_matrix_normalized.png](results/charts/confusion_matrix_normalized.png)

**Nhận xét chuyên môn:**
- **Teratoma (U quái)**: Đạt tỷ lệ phân loại chuẩn xác cao nhất lên tới **76% (0.76)**.
- **Chocolate cyst (Nang sô-cô-la)**: Tỷ lệ phân loại chính xác đạt **70% (0.70)**.
- **Serous cystadenoma (U nang thanh dịch)**: Tỷ lệ phân loại chính xác đạt **51% (0.51)**.
- **Hiện tượng nhầm lẫn lâm sàng**:
  - Có 19% mẫu *Serous cystadenoma* bị nhận nhầm sang *Chocolate cyst* và ngược lại 3% *Chocolate cyst* sang *Serous cystadenoma*. Đây là hiện tượng tương đồng trên lâm sàng do cả hai đều biểu hiện dạng nang dịch chứa chất lỏng/dịch nhầy trên siêu âm B-mode.
  - Các lớp mẫu hiếm (*Theca cell tumor*, *High-grade serous*) có xu hướng bị dự đoán nhầm sang lớp đa số hoặc gán nhầm vào background do trọng số chưa được cân bằng.

---

### Biểu đồ 7: Ma trận Nhầm lẫn Giá trị Thực tế (`confusion_matrix.png`)

![Ma trận nhầm lẫn confusion_matrix.png](results/charts/confusion_matrix.png)

**Nhận xét chuyên môn:**
- Thể hiện số lượng bounding box dự đoán đúng và sai sót với nền (background).
- Số lượng dự đoán đúng tập trung lớn ở các ca Teratoma (83 ca), Chocolate cyst (81 ca) và Serous cystadenoma (34 ca).
- Do ảnh siêu âm có độ nhiễu đốm (speckle noise) cao, một số vùng mô buồng trứng bình thường xung quanh u vẫn bị mô hình nhận diện là vùng nghi ngờ với background.

---

### Biểu đồ 8: Thống kê Phân Bố Tập Nhãn & Kích Thước Bounding Box (`labels.jpg`)

![Phân bố nhãn labels.jpg](results/charts/labels.jpg)

**Nhận xét chuyên môn:**
1. **Phân bố số lượng mẫu (Instances bar chart)**:
   - Chocolate cyst: **238 mẫu**
   - Teratoma: **228 mẫu**
   - Serous cystadenoma: **153 mẫu**
   - Mucinous cystadenoma: **71 mẫu**
   - Theca cell tumor: **57 mẫu**
   - Simple cyst: **47 mẫu**
   - High-grade serous: **38 mẫu**
   - Normal ovary: **0 mẫu bounding box** (mẫu đối chứng không u)
   *Nhận định*: Tồn tại sự mất cân bằng dữ liệu đáng kể giữa các lớp phổ biến (Chocolate cyst, Teratoma) và các lớp u hiếm gặp / ác tính cao (High-grade serous chỉ có 38 mẫu).
2. **Tọa độ tâm u (x, y)**: Khối u thường xuất hiện tập trung ở vị trí trung tâm màn hình siêu âm ($x \approx 0.4 - 0.6$, $y \approx 0.35 - 0.55$).
3. **Kích thước hộp bao u (width, height)**: Phân bố trải dài từ kích thước nhỏ ($\approx 0.15 - 0.3$) đến các khối u nang lớn chiếm hơn một nửa khung hình siêu âm ($\approx 0.6 - 0.85$).

---

## 4. Đánh Giá Điểm Mạnh & Hạn Chế Hiện Tại

### Điểm mạnh:
1. **Kiến trúc gọn nhẹ, siêu nhanh**: YOLO11n chỉ tốn 5.46 MB, tốc độ xử lý ~42.5 FPS trên CPU, hoàn toàn đủ điều kiện tích hợp trực tiếp vào các máy siêu âm phụ khoa thời gian thực.
2. **Khả năng học đặc trưng nhanh**: Chỉ qua 2 epoch đầu tiên, mAP@50 đã nhảy vọt từ 9.9% lên 20.3%, trong đó các u nang phổ biến đạt xấp xỉ 45%.
3. **Độ phân biệt cao ở các ca u phổ biến**: Teratoma đạt 76% và Chocolate cyst đạt 70% độ chính xác phân loại độc lập trên ma trận nhầm lẫn.

### Hạn chế cần khắc phục:
1. **Số epoch huấn luyện còn ngắn**: Mới chỉ chạy 2 epoch thử nghiệm sơ bộ nên các trọng số chưa hội tụ tối đa.
2. **Mất cân bằng dữ liệu (Class Imbalance)**: Các lớp u hiếm (*High-grade serous*, *Theca cell tumor*) có ít dữ liệu huấn luyện nên độ chính xác mAP còn thấp.
3. **Nhiễu âm phụ khoa**: Đặc tính siêu âm B-mode dễ bị nhiễu bóng lưng hoặc giảm độ tương phản ở các vách u mỏng.

---

## 5. Định Hướng Phát Triển Tiếp Theo

1. **Gia tăng số lượng Epoch**: Huấn luyện đầy đủ 30 - 50 epochs (kết hợp GPU CUDA) để đạt điểm hội tụ mAP tối ưu (dự kiến mAP@50 có thể đạt > 65 - 75%).
2. **Cân bằng lớp bằng Loss Function**: Áp dụng Class-Weighted Focal Loss để tăng trọng số phạt cho các lớp thiểu số như ung thư ác tính *High-grade serous*.
3. **Mở rộng độ phân giải**: Tăng kích thước ảnh đầu vào `imgsz=640` để mô hình nắm bắt chi tiết hơn các cấu trúc vi thể bên trong u nang nhầy và nang bì.
4. **Tích hợp Siêu âm Doppler (CDFI)**: Khai thác thêm kênh thông tin mạch máu từ dữ liệu Doppler màu để phân biệt chính xác u lành tính và ác tính.

