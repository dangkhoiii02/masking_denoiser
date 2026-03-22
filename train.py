import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# --- 1. CẤU HÌNH ---
DATASET_PATH = 'dataset'
SAMPLE_RATE = 16000
DURATION = 1.0 
SAMPLES = int(SAMPLE_RATE * DURATION)
N_FFT = 512
HOP_LENGTH = 160 

# --- 2. XỬ LÝ TÍN HIỆU THEO PHƯƠNG PHÁP MASKING ---
def process_audio_pair(clean_path, noisy_path):
    # Load âm thanh
    clean, _ = librosa.load(clean_path, sr=SAMPLE_RATE, duration=DURATION)
    noisy, _ = librosa.load(noisy_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Pad nếu thiếu
    if len(clean) < SAMPLES: clean = np.pad(clean, (0, SAMPLES - len(clean)))
    if len(noisy) < SAMPLES: noisy = np.pad(noisy, (0, SAMPLES - len(noisy)))

    # Tính STFT
    stft_clean = librosa.stft(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_noisy = librosa.stft(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    mag_clean = np.abs(stft_clean)
    mag_noisy = np.abs(stft_noisy)

    # ---------------------------------------------------------
    # BƯỚC ĐỘT PHÁ: TÍNH TOÁN "MẶT NẠ" (IDEAL RATIO MASK)
    # Tỷ lệ giữa tín hiệu sạch / tín hiệu nhiễu
    # ---------------------------------------------------------
    mask_target = mag_clean / (mag_noisy + 1e-8) # Cộng 1e-8 để tránh chia cho 0
    mask_target = np.clip(mask_target, 0, 1)     # Ép giá trị về khoảng [0, 1]

    # ĐẦU VÀO CỦA AI (X): Ảnh phổ nhiễu (chuyển sang dB để dễ học)
    mag_noisy_db = librosa.amplitude_to_db(mag_noisy)
    X_input = (mag_noisy_db + 80) / 80 
    X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

    # ĐẦU RA CỦA AI (Y): Chiếc mặt nạ (Mask)
    Y_target = mask_target.reshape(mask_target.shape[0], mask_target.shape[1], 1)

    return X_input, Y_target

# --- 3. LOAD DATASET ---
def load_data(limit=2000):
    X, Y = [], []
    clean_dir = os.path.join(DATASET_PATH, 'clean')
    noisy_dir = os.path.join(DATASET_PATH, 'noisy')
    files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
    print(f"Đang chuẩn bị {min(len(files), limit)} Mặt nạ (Masks)...")
    
    for f in files[:limit]:
        try:
            x_in, y_tar = process_audio_pair(os.path.join(clean_dir, f), os.path.join(noisy_dir, f))
            X.append(x_in)
            Y.append(y_tar)
        except: continue
    return np.array(X), np.array(Y)

# --- 4. KIẾN TRÚC U-NET (DỰ ĐOÁN MẶT NẠ) ---
def build_mask_unet(input_shape):
    inputs = Input(shape=input_shape)

    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2), padding='same')(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    
    u1 = layers.UpSampling2D((2, 2))(c2)
    u1 = layers.Cropping2D(cropping=((0, 1), (0, 1)))(u1)
    merge1 = layers.Concatenate()([c1, u1])
    
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge1)

    # Lớp xuất ra Mặt nạ (Sigmoid đảm bảo xuất ra đúng giá trị 0 đến 1)
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(c3)

    model = models.Model(inputs, outputs)
    
    # Dùng Binary Crossentropy vì bản chất mặt nạ giống như phân loại pixel (Nhiễu = 0, Sạch = 1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

if __name__ == "__main__":
    X_train, Y_train = load_data(limit=2000)
    model = build_mask_unet(X_train.shape[1:])
    
    print("\nBắt đầu huấn luyện AI để tìm Mặt nạ...")
    model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('masking_denoiser.tflite', 'wb') as f:
        f.write(tflite_model)
    print("\nHoàn thành! Model học Mặt Nạ đã sẵn sàng.")