# Masking Denoiser for ESP32-S3

Hệ thống lọc nhiễu âm thanh thời gian thực sử dụng kỹ thuật **Masking (Ideal Ratio Mask)** và mô hình **CNN U-Net**, được tối ưu hóa để chạy trên vi điều khiển ESP32-S3 (TensorFlow Lite).

## 🚀 Tính năng nổi bật
- **Kỹ thuật Masking chuyên sâu**: Không chỉ lọc nhiễu đơn thuần, AI học cách tạo ra một "mặt nạ" để giữ lại giọng nói và triệt tiêu tạp âm.
- **Tối ưu cho ESP32-S3**: Sử dụng TFLite với kỹ thuật chuẩn hóa dữ liệu dB giúp mô hình nhẹ nhưng vẫn đạt hiệu suất cao.
- **Auto-Gain Control**: Tự động bù lại âm lượng sau khi lọc để giọng nói luôn rõ ràng.
- **Dynamic SNR Training**: Dataset được gen ngẫu nhiên với SNR từ -5dB đến 15dB, giúp AI làm việc tốt trong nhiều môi trường khác nhau.

## 🛠️ Yêu cầu hệ thống
- Python 3.8+
- Các thư viện: `numpy`, `librosa`, `tensorflow`, `soundfile`, `sounddevice`.

Cài đặt nhanh:
```bash
pip install -r requirements.txt
```

## 📂 Cấu trúc dự án
- `prepare_data.py`: Script tạo bộ dữ liệu huấn luyện (Dataset) từ file sạch (LibriSpeech) và file nhiễu (ESC-50).
- `train.py`: Xây dựng kiến trúc U-Net, huấn luyện mô hình dự đoán Mask và xuất file `.tflite`.
- `test.py`: Kiểm tra khả năng lọc nhiễu của mô hình trên file âm thanh thực tế, kèm theo tính năng phát âm thanh so sánh.
- `masking_denoiser.tflite`: Mô hình AI đã được huấn luyện sẵn.

## 📖 Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Chạy script sau để tạo bộ dataset mẫu:
```bash
python prepare_data.py
```
*Lưu ý: Bạn cần có thư mục `LibriSpeech` và `ESC-50-master` hoặc sửa đường dẫn trong code.*

### 2. Huấn luyện mô hình
Để tự huấn luyện lại mô hình:
```bash
python train.py
```
Kết quả sẽ tạo ra file `masking_denoiser.tflite`.

### 3. Kiểm tra kết quả
Chạy script test để nghe sự khác biệt:
```bash
python test.py
```

## 🧠 Nguyên lý hoạt động
1. **STFT**: Chuyển tín hiệu âm thanh từ miền thời gian sang miền tần số.
2. **Feature Extraction**: Chuyển biên độ sang thang đo log (dB) và chuẩn hóa về khoảng [0, 1].
3. **U-Net Prediction**: AI dự đoán `Ideal Ratio Mask` (Mặt nạ tỷ lệ lý tưởng).
4. **Mask Application**: Nhân trực tiếp mặt nạ vào phổ biên độ của âm thanh nhiễu.
5. **iSTFT**: Chuyển phổ đã lọc về lại miền thời gian để phát ra loa.

---
*Dự án được phát triển cho mục đích nghiên cứu hệ thống nhúng và AI Edge Computing.*
