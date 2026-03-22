import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import time
import os

# --- 1. CẤU HÌNH THÔNG SỐ ---
MODEL_PATH = 'masking_denoiser.tflite'
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)

def test_masking_pro(noisy_file_path):
    if not os.path.exists(noisy_file_path):
        print(f"Lỗi: Không tìm thấy file {noisy_file_path}")
        return

    print(f"\n--- Đang xử lý file: {noisy_file_path} ---")
    
    # 1. LOAD MODEL (TFLITE)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. ĐỌC ÂM THANH & TRÍCH XUẤT ĐẶC TRƯNG (STFT)
    audio, _ = librosa.load(noisy_file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    # Biến đổi Fourier để lấy Biên độ (Magnitude) và Pha (Phase)
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, phase = librosa.magphase(stft)

    # Chuẩn hóa đầu vào (dB) để AI dễ đọc
    mag_db = librosa.amplitude_to_db(magnitude)
    X_input = (mag_db + 80) / 80
    X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1], 1).astype(np.float32)

    # 3. AI DỰ ĐOÁN MẶT NẠ (INFERENCE)
    print("🤖 AI đang phân tích và tạo Mặt nạ lọc nhiễu...")
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], X_input)
    interpreter.invoke()
    predicted_mask = interpreter.get_tensor(output_details[0]['index'])
    print(f"⏱️ Thời gian xử lý (Mac M1): {(time.time() - start_time)*1000:.2f} ms")

    # 4. LỌC NHIỄU CỨNG (HARD THRESHOLDING)
    predicted_mask = predicted_mask.reshape(magnitude.shape)
    
    # Tinh chỉnh: Ngưỡng 0.5 là mức cân bằng hoàn hảo khi đã có Auto-Gain.
    # Nó đủ để cắt tạp âm mà không "chém" mất các âm gió của giọng nói.
    THRESHOLD = 0.5
    KEEP_VOL = 1.0
    SUPPRESS_VOL = 0.01
    
    # Tạo mặt nạ sắc nét
    sharp_mask = np.where(predicted_mask > THRESHOLD, KEEP_VOL, SUPPRESS_VOL)
    
    # Nhân ma trận: Tạp âm biến mất, giọng nói giữ nguyên
    clean_magnitude = magnitude * sharp_mask

    # 5. TÁI TẠO ÂM THANH (iSTFT)
    clean_stft = clean_magnitude * phase
    clean_audio = librosa.istft(clean_stft, hop_length=HOP_LENGTH)

    # ======================================================
    # 6. AUTO-GAIN (TỰ ĐỘNG BÙ ÂM LƯỢNG TIẾNG NGƯỜI)
    # ======================================================
    max_original_vol = np.max(np.abs(audio))
    max_clean_vol = np.max(np.abs(clean_audio))
    
    if max_clean_vol > 0:
        gain_factor = max_original_vol / max_clean_vol
        clean_audio = clean_audio * gain_factor # Khuếch đại lên đúng bằng mức gốc

    # Ép giới hạn an toàn để không bị rè loa (Clipping)
    clean_audio = np.clip(clean_audio, -1.0, 1.0)

    # 7. PHÁT RA LOA
    print("\n🔊 Đang phát âm thanh GỐC (Có tạp âm)...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    
    time.sleep(1)
    
    print("🔊 Đang phát âm thanh ĐÃ LỌC (Sạch & Âm lượng chuẩn)...")
    sd.play(clean_audio, SAMPLE_RATE)
    sd.wait()
    print("\n✅ --- Hoàn thành Test! ---")

# --- CHẠY THỬ ---
# (Nhớ đổi thành đường dẫn file nhiễu của bạn nhé)
test_file = 'dataset/noisy/pro_sample_40.wav' 
test_masking_pro(test_file)