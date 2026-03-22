import os
import librosa
import soundfile as sf
import numpy as np
import random

# Cấu hình đường dẫn
CLEAN_DIR = 'LibriSpeech/dev-clean' 
NOISE_DIR = 'ESC-50-master/audio'
OUTPUT_CLEAN = 'dataset/clean'
OUTPUT_NOISY = 'dataset/noisy'

# Thông số chuẩn cho ESP32-S3
SR = 16000 
DURATION = 1.0  # Giảm xuống 1s để model chạy nhanh và Real-time tốt hơn
TARGET_LEN = int(SR * DURATION)
LIMIT = 2000    # Tăng lên 2000 mẫu cho "xịn"

def create_folders():
    os.makedirs(OUTPUT_CLEAN, exist_ok=True)
    os.makedirs(OUTPUT_NOISY, exist_ok=True)

def get_rms(y):
    """Tính mức năng lượng trung bình của tín hiệu"""
    return np.sqrt(np.mean(y**2))

def mix_audio_pro():
    create_folders()
    
    # Lấy danh sách file
    clean_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(CLEAN_DIR) 
                   for f in filenames if f.endswith('.flac') or f.endswith('.wav')]
    noise_files = [os.path.join(NOISE_DIR, f) for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]
    
    random.shuffle(clean_files) # Trộn ngẫu nhiên danh sách gốc
    
    count = 0
    print(f"Bắt đầu gen {LIMIT} mẫu dataset chất lượng cao...")

    while count < LIMIT:
        try:
            clean_path = random.choice(clean_files)
            clean, _ = librosa.load(clean_path, sr=SR, mono=True)
            
            # Chỉ lấy các đoạn có tiếng người (bỏ qua đoạn lặng đầu/cuối)
            clean, _ = librosa.effects.trim(clean)
            
            if len(clean) < TARGET_LEN: continue
            
            # Cắt một đoạn ngẫu nhiên 1 giây trong file tiếng người
            start_idx = random.randint(0, len(clean) - TARGET_LEN)
            clean_segment = clean[start_idx:start_idx + TARGET_LEN]

            # Chọn nhiễu ngẫu nhiên
            noise_path = random.choice(noise_files)
            noise, _ = librosa.load(noise_path, sr=SR, mono=True)
            
            # Cắt hoặc lặp lại nhiễu cho đủ 1 giây
            if len(noise) < TARGET_LEN:
                noise_segment = np.tile(noise, int(np.ceil(TARGET_LEN/len(noise))))[:TARGET_LEN]
            else:
                start_noise = random.randint(0, len(noise) - TARGET_LEN)
                noise_segment = noise[start_noise:start_noise + TARGET_LEN]

            # --- BƯỚC QUAN TRỌNG: TÍNH TOÁN SNR ---
            # Chọn ngẫu nhiên mức độ nhiễu từ -5dB (nhiễu to hơn người) đến 15dB (người to hơn nhiễu)
            snr_db = random.uniform(-5, 15)
            
            rms_clean = get_rms(clean_segment)
            rms_noise = get_rms(noise_segment)
            
            if rms_noise == 0: continue # Tránh lỗi chia cho 0
            
            # Tính toán mức năng lượng nhiễu cần thiết để đạt SNR mục tiêu
            rms_noise_target = rms_clean / (10**(snr_db/20))
            
            # Điều chỉnh âm lượng nhiễu
            noise_segment = noise_segment * (rms_noise_target / rms_noise)
            
            # Trộn
            noisy_segment = clean_segment + noise_segment
            
            # Chuẩn hóa biên độ (Peak Normalize) về -1.0 đến 1.0 để không bị rè (clipping)
            max_val = np.max(np.abs(noisy_segment))
            if max_val > 1.0:
                noisy_segment = noisy_segment / max_val
                clean_segment = clean_segment / max_val # Sạch cũng phải giảm theo để khớp tỉ lệ

            # Lưu file
            file_name = f"pro_sample_{count}.wav"
            sf.write(os.path.join(OUTPUT_CLEAN, file_name), clean_segment, SR, subtype='PCM_16')
            sf.write(os.path.join(OUTPUT_NOISY, file_name), noisy_segment, SR, subtype='PCM_16')
            
            count += 1
            if count % 100 == 0:
                print(f"Đã hoàn thành: {count}/{LIMIT}")

        except Exception as e:
            continue

if __name__ == "__main__":
    mix_audio_pro()
    print("XONG! Bạn đã có bộ Dataset 'xịn' với Dynamic SNR.")