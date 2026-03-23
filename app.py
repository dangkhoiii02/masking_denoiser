import serial
import threading
import asyncio
import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import tensorflow as tf
import librosa
import os

app = FastAPI()

SERIAL_PORT = '/dev/cu.usbmodem21201' 
BAUD_RATE = 921600 # Khuyến nghị đổi thành 921600 ở cả ESP32 và đây để chạy Real-time
SAMPLE_RATE = 16000

CHUNK_SIZE = 16000  # ĐÚNG 1 GIÂY ÂM THANH CHO MODEL CỦA BẠN
N_FFT = 512
HOP_LENGTH = 160

# ==========================================
# 🧠 KHỞI TẠO TFLITE MODEL
# ==========================================
MODEL_LOADED = False
print("⏳ Đang khởi động hệ thống và tải mô hình AI...")

try:
    interpreter = tf.lite.Interpreter(model_path="masking_denoiser.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Tải mô hình thành công! Đang chờ đủ {CHUNK_SIZE} mẫu âm thanh từ ESP32...")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Cảnh báo: Lỗi tải Model TFLite ({e}). Server sẽ chạy mà không có AI.")

audio_queue = []
ui_data_package = None
lock = threading.Lock()

# ==========================================
# 🧠 HÀM SUY LUẬN AI (ĐÃ THÊM TIỀN XỬ LÝ CHO MIC THẬT)
# ==========================================
def run_tflite_model(raw_chunk):
    if not MODEL_LOADED:
        return raw_chunk 
        
    try:
        # 1. Chuyển sang float32
        audio = raw_chunk.astype(np.float32) / 32768.0
        
        # --- BƯỚC QUAN TRỌNG ĐỂ MIC THẬT CHẠY ĐƯỢC AI ---
        # 1.1. Loại bỏ DC Offset (Nhiễu nền tĩnh của phần cứng)
        audio = audio - np.mean(audio) 
        # 1.2. Chuẩn hóa âm lượng (Giống với hàm librosa.load lúc train)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val 
        # ------------------------------------------------

        # 2. STFT & CHUẨN HÓA
        stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        magnitude, phase = librosa.magphase(stft)

        mag_db = librosa.amplitude_to_db(magnitude)
        X_input = (mag_db + 80) / 80
        X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1], 1).astype(np.float32)

        # 3. AI DỰ ĐOÁN MẶT NẠ
        interpreter.set_tensor(input_details[0]['index'], X_input)
        interpreter.invoke()
        predicted_mask = interpreter.get_tensor(output_details[0]['index'])
        
        # Bỏ comment dòng print dưới đây để xem AI có đang hoạt động không
        # print(f"👉 Trung bình mặt nạ: {np.mean(predicted_mask):.2f} (Càng gần 0 lọc càng mạnh)")

        # 4. LỌC NHIỄU CỨNG
        predicted_mask = predicted_mask.reshape(magnitude.shape)
        
        # TĂNG NGƯỠNG LỌC (Từ 0.5 lên 0.7) ĐỂ CẮT NHIỄU MẠNH TAY HƠN
        sharp_mask = np.where(predicted_mask > 0.7, 1.0, 0.01) 
        clean_magnitude = magnitude * sharp_mask

        # 5. TÁI TẠO ÂM THANH (iSTFT)
        clean_stft = clean_magnitude * phase
        clean_audio = librosa.istft(clean_stft, hop_length=HOP_LENGTH)

        # Cân bằng độ dài
        if len(clean_audio) > len(audio):
            clean_audio = clean_audio[:len(audio)]
        elif len(clean_audio) < len(audio):
            clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))

        # 6. AUTO-GAIN
        max_original_vol = np.max(np.abs(audio))
        max_clean_vol = np.max(np.abs(clean_audio))
        if max_clean_vol > 0:
            gain_factor = max_original_vol / max_clean_vol
            clean_audio = clean_audio * gain_factor 

        clean_audio = np.clip(clean_audio, -1.0, 1.0)
        return np.clip(clean_audio * 32768.0, -32768, 32767).astype(np.int16)

    except Exception as e:
        print(f"❌ Lỗi xử lý AI: {e}")
        return raw_chunk
# ==========================================
# LUỒNG ĐỌC USB VÀ PHÁT LOA (ĐÃ TỐI ƯU CHỐNG LẶP)
# ==========================================
def process_audio_stream():
    global ui_data_package
    BYTES_PER_CHUNK = CHUNK_SIZE * 2 
    
    try:
        speaker = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        speaker.start()
    except Exception as e:
        print(f"⚠️ Lỗi loa: {e}")
        speaker = None
    
    try:
        ser = serial.Serial(SERIAL_PORT, 921600, timeout=1)
        # DỌN SẠCH BUFFER KHI VỪA MỞ (Chống phát tiếng cũ)
        ser.reset_input_buffer() 
        print(f"✅ Đã kết nối ESP32 - Hệ thống đã sẵn sàng!")
        
        audio_buffer = bytearray()
        
        while True:
            if ser.in_waiting > 0:
                raw_bytes = ser.read(ser.in_waiting)
                audio_buffer.extend(raw_bytes)
                
                # Nếu bộ đệm bị dồn ứ quá nhiều (do AI xử lý chậm), hãy bỏ bớt dữ liệu cũ
                if len(audio_buffer) > BYTES_PER_CHUNK * 3:
                    print("⚠️ Cảnh báo: Trễ dữ liệu, đang dọn bộ đệm...")
                    audio_buffer = audio_buffer[-BYTES_PER_CHUNK:]

                if len(audio_buffer) >= BYTES_PER_CHUNK:
                    chunk_bytes = audio_buffer[:BYTES_PER_CHUNK]
                    del audio_buffer[:BYTES_PER_CHUNK]
                    
                    raw_chunk = np.frombuffer(chunk_bytes, dtype=np.int16).copy()
                    
                    # --- KIỂM TRA NGƯỠNG ÂM THANH (NOISE GATE) ---
                    # Tính độ lớn trung bình của âm thanh (RMS)
                    rms = np.sqrt(np.mean(raw_chunk.astype(np.float32)**2))
                    
                    # Nếu tiếng quá nhỏ (dưới mức 100), coi như im lặng và bỏ qua
                    if rms < 100:
                        clean_chunk = np.zeros_like(raw_chunk) # Trả về mảng 0 (im lặng)
                    else:
                        # CHỈ CHẠY AI KHI CÓ TIẾNG NÓI ĐỦ TO
                        clean_chunk = run_tflite_model(raw_chunk)
                    
                    if speaker:
                        speaker.write(clean_chunk)
                    
                    # Cập nhật UI
                    fft_result = np.abs(np.fft.rfft(clean_chunk[:1024]))
                    fft_scaled = (fft_result / 100).tolist()
                    with lock:
                        ui_data_package = {
                            "raw_wave": raw_chunk[::16].tolist(),     
                            "clean_wave": clean_chunk[::16].tolist(), 
                            "fft": fft_scaled[:128]
                        }
                        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        if speaker: speaker.stop()

# Khởi chạy luồng
threading.Thread(target=process_audio_stream, daemon=True).start()

# Khởi chạy luồng phần cứng ngầm (KHÔNG CHẶN SERVER)
threading.Thread(target=process_audio_stream, daemon=True).start()

# ==========================================
# SERVER FASTAPI CHO GIAO DIỆN
# ==========================================
@app.get("/")
async def serve_ui():
    return FileResponse('index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global ui_data_package
    try:
        while True:
            with lock:
                data_to_send = ui_data_package
            if data_to_send:
                await websocket.send_json(data_to_send)
                # Đã gửi xong thì xóa để chờ 1 giây sau có dữ liệu mới mới gửi tiếp
                with lock:
                    ui_data_package = None 
            await asyncio.sleep(0.05) 
    except Exception:
        pass

if __name__ == "__main__":
    print("🚀 Bắt đầu mở Web Server UI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)