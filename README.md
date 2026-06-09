# DeepFoot — Phân loại Cầu thủ / Thủ môn / Trọng tài từ ảnh bóng đá (Deep Learning)

> Đồ án môn **Mạng nơ-ron và Học sâu (Neural Networks & Deep Learning)** — Chương trình
> Thạc sĩ Công nghệ Thông tin, Trường Đại học Công nghệ TP.HCM (HUTECH).

Bài toán **phân loại ảnh 3 lớp** — `player` (cầu thủ), `goalkeeper` (thủ môn),
`referee` (trọng tài) — từ hình ảnh bóng đá, sử dụng **Convolutional Neural Network (CNN)**
huấn luyện từ đầu, có so sánh với các mô hình **Transfer Learning** phổ biến.

## Quy trình thực nghiệm

```
Xác định mục tiêu & dataset → Tiền xử lý dữ liệu → Xây dựng mô hình CNN
        → Cấu hình huấn luyện → Huấn luyện → Đánh giá → Tinh chỉnh → Cải thiện hiệu suất
```

## Dữ liệu

- 3 lớp cân bằng: `player`, `goalkeeper`, `referee`.
- Train ≈ **2.385 ảnh** (referee 800 · goalkeeper 794 · player 791).
- Test ≈ **601 ảnh** (player 202 · goalkeeper 201 · referee 198).
- Ảnh đầu vào chuẩn hóa về **224 × 224 × 3**; nạp qua `ImageDataGenerator`
  (`train_gen` / `valid_gen` / `test_gen`).

## Kiến trúc mô hình (Custom CNN — cảm hứng AlexNet)

`Conv2D → BatchNormalization → MaxPooling` (lặp) → `Flatten` →
`Dense(4096) → Dropout → Dense(4096) → Dropout → Dense(3, softmax)`

- Tổng tham số: **≈ 46.76M** (~178 MB); trainable ≈ 46.76M.
- Chống overfitting bằng `BatchNormalization` + `Dropout`.

### Early Stopping
```python
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=30,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
)
```
Theo dõi hiệu suất trên tập validation; dừng sớm khi không cải thiện sau `patience` epoch,
giữ lại trọng số tốt nhất → tiết kiệm thời gian, giảm overfitting.

## Kết quả

| Tập | Accuracy | Loss |
|-----|----------|------|
| Train | 0.9768 | 0.0773 |
| Validation | 0.9329 | 0.2596 |
| Test | 0.9329 | 0.2596 |

**Per-class (validation):**

| Lớp | Precision | Recall | F1-Score |
|-----|-----------|--------|----------|
| Goalkeeper | 0.9208 | 0.9254 | 0.9231 |
| Player | 0.9394 | 0.9208 | 0.9300 |
| Referee | 0.9751 | 0.9899 | 0.9824 |

> Nhầm lẫn chủ yếu giữa **goalkeeper ↔ player** (trang phục dễ giống); `referee` phân loại tốt nhất.

## So sánh Transfer Learning

Đối chiếu Custom CNN với **MobileNetV2, ResNet50, EfficientNetB3, VGG16, YOLOv8n-cls**
(validation accuracy/loss theo epoch):

- Tốt nhất: **Custom CNN** và **YOLOv8n-cls** (~0.93–0.95).
- Trung bình: MobileNetV2 (~0.85), VGG16 (~0.78).
- Kém trên dataset này: ResNet50 (~0.44), EfficientNetB3 (~0.33).

## Công nghệ

Python · TensorFlow/Keras (`Conv2D`, `BatchNormalization`, `Dropout`, `EarlyStopping`) ·
NumPy · Matplotlib / Seaborn (biểu đồ accuracy/loss, confusion matrix) · Jupyter/Colab.

## Cài đặt & chạy

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
# Mở notebook huấn luyện (Colab khuyến nghị có GPU) và chạy lần lượt các cell
```

> Cập nhật đường dẫn dataset (thư mục `train/`, `valid/`, `test/` theo từng lớp) cho khớp repo.

## Hướng phát triển — DeepFoot (pipeline đầy đủ)

Mở rộng từ bài phân loại đơn lẻ thành pipeline phân tích trận đấu:

```
Video Stitching → Object Detection (SAHI + YOLOv8) → Object Tracking (ByteTrack)
   → Image Classification / Clustering (team A/B, GK, referee)
   → Action & Event Recognition (MobileViT) → Post-processing → Player/Team stats
```

- Mở rộng dataset, thêm bước vào pipeline.
- Tự dò siêu tham số bằng **GridSearch / RandomSearch**; ensemble nhiều mô hình transfer learning.
- Thử kiến trúc mới (**EfficientNetV2, MobileViT**).
- Trích xuất **thống kê cầu thủ/đội** (số cú sút, đường chuyền, kiểm soát bóng…).
- Phát triển giao diện + **deploy mô hình thành API**.

## Nhóm thực hiện

Nguyễn Ngọc Đỉnh · Hà Anh Dũng · Nguyễn Minh Trung Nghĩa
GVHD: TS. Nguyễn Thị Hải Bình
