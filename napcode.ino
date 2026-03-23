#include <driver/i2s.h>

#define I2S_WS 15  
#define I2S_SCK 14 
#define I2S_SD 13  
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000 
#define DMA_BUF_COUNT 8
#define DMA_BUF_LEN 1024

void setup() {
  // BẮT BUỘC TĂNG TỐC ĐỘ LÊN MỨC CAO NHẤT
  Serial.begin(921600); 
  delay(2000); 

  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = i2s_bits_per_sample_t(16),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = DMA_BUF_COUNT,
    .dma_buf_len = DMA_BUF_LEN,
    .use_apll = false
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);
}

void loop() {
  size_t bytesIn = 0;
  char buffer[DMA_BUF_LEN]; // Dùng mảng char (byte) thay vì int16
  
  esp_err_t result = i2s_read(I2S_PORT, &buffer, DMA_BUF_LEN, &bytesIn, portMAX_DELAY);

  if (result == ESP_OK && bytesIn > 0) {
    // Truyền thẳng cục nhị phân qua cáp USB, không tốn thời gian dịch ra chữ nữa
    Serial.write((uint8_t*)buffer, bytesIn);
  }
}