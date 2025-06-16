import numpy as np
import matplotlib.pyplot as plt

# Tạo tín hiệu mẫu: tổng 2 sóng sin với tần số khác nhau
t = np.linspace(0, 1, 400, endpoint=False)
freq1 = 5  # tần số sóng 1
freq2 = 20  # tần số sóng 2
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# Tính FFT
fft_result = np.fft.fft(signal)

# Tính tần số tương ứng
n = len(signal)
freq = np.fft.fftfreq(n, d=t[1] - t[0])

# Lấy phần biên độ (magnitude) của FFT, chỉ lấy nửa phổ do đối xứng
magnitude = np.abs(fft_result)[:n // 2]
freq = freq[:n // 2]

# Vẽ tín hiệu gốc
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Vẽ phổ tần số (FFT)
plt.subplot(1, 2, 2)
plt.stem(freq, magnitude)
plt.title('FFT Magnitude Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()
