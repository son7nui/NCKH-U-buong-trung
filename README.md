# Ứng Dụng Học Sâu Cho Phát Hiện Khối U Buồng Trứng Từ Ảnh Siêu Âm
*Deep Learning for Ovarian Tumor Detection from Ultrasound Images (MMOTU Dataset)*

Đề tài Nghiên cứu Khoa học (NCKH) phát triển và ứng dụng các mô hình học sâu thị giác máy tính tiên tiến nhằm hỗ trợ phát hiện, định vị và phân loại đa lớp các khối u buồng trứng từ hình ảnh siêu âm phụ khoa theo thời gian thực.

---

## 1. Tổng quan Bộ dữ liệu (Dataset MMOTU)

Bộ dữ liệu **MMOTU** (*Multi-Modality Ovarian Tumor Ultrasound*) gồm 2 tập con chính:
- **`OTU_2d`**: 1.469 ảnh siêu âm 2D (B-mode và Color Doppler CDFI).
- **`OTU_3d`**: 170 ảnh siêu âm cản âm vi mạch (CEUS).

### 8 Nhóm Nhãn Bệnh Học (Fine-grained Classes):
- **0 - Chocolate cyst**: U nang sô-cô-la (Nang lạc nội mạc tử cung)
- **1 - Serous cystadenoma**: U nang thanh dịch
- **2 - Teratoma**: U quái buồng trứng (Nang bì)
- **3 - Theca cell tumor**: U tế bào vỏ
- **4 - Simple cyst**: U nang đơn giản (Nang cơ năng)
- **5 - Normal ovary**: Buồng trứng bình thường (Mẫu đối chứng âm bản)
- **6 - Mucinous cystadenoma**: U nang nhầy
- **7 - High-grade serous**: Ung thư biểu mô thanh dịch độ ác tính cao

---

## 2. Cấu trúc Dự án

```text
NCKH/
├── MMOTU/                      # Dataset gốc (OTU_2d & OTU_3d)
├── dataset_yolo/               # Dữ liệu chuẩn hóa YOLO (Images & Labels)
│   ├── images/ (train, val)
│   ├── labels/ (train, val)
│   └── dataset.yaml            # Cấu hình dataset YOLO
├── src/
│   ├── prepare_yolo_dataset.py # Script trích xuất bounding box từ mask
│   ├── train_yolo.py           # Huấn luyện mô hình YOLO11n
│   └── evaluate_and_visualize.py # Đánh giá & trực quan hóa đối đầu nhãn
├── runs/detect/                # Trọng số best.pt, đồ thị PR, Confusion matrix
├── results/visualizations/     # Ảnh trực quan hóa Ground Truth vs Prediction
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 3. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### Bước 1: Tiền xử lý & Trích xuất Bounding Box từ Mask
```bash
python src/prepare_yolo_dataset.py
```

### Bước 2: Huấn luyện Mô hình YOLO11
```bash
python src/train_yolo.py --epochs 50 --batch 16 --imgsz 640
```

### Bước 3: Đánh giá & Xuất ảnh Trực quan hóa
```bash
python src/evaluate_and_visualize.py --weights runs/detect/runs/detect/ovarian_tumor_yolo11n/weights/best.pt --samples 15
```

---

## 4. Kết quả Thực nghiệm Ban đầu

- **Kiến trúc mô hình**: YOLO11n (2,59 triệu tham số, 6.5 GFLOPs).
- **Tốc độ suy luận**: **23.5 ms/ảnh (~42 FPS)** trên CPU (đáp ứng tiêu chuẩn thời gian thực lâm sàng).
- **Độ chính xác phát hiện mAP@50**:
  - Serous cystadenoma: **44.9%**
  - Chocolate cyst: **39.8%**
  - Mucinous cystadenoma: **21.1%**
  - Teratoma: **20.7%**
- **Trọng số mô hình**: Lưu tại `runs/detect/runs/detect/ovarian_tumor_yolo11n/weights/best.pt`.

