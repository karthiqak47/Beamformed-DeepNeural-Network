import numpy as np
import matplotlib.pyplot as plt
import random

# --- Parameters ---
T = 200                     # Number of time steps (frames)
c = 1500                    # Speed of sound (m/s)
fc = 1000                   # Center frequency (Hz)
fs = 100                    # Sampling rate (frames per second = temporal resolution)
M = 32                      # Number of hydrophones
d = 0.45 * c / fc           # Element spacing (half-wavelength)
Theta = np.arange(0, 181)   # Beam angles
doa = 28                   # True DOA for synthetic target
snr_min = -14               # Minimum SNR (dB)
signal_band = 5            # Frequency bins to keep around fc (±Hz)

# --- Buffers ---
labels = []
Z = np.zeros((T, len(Theta)), dtype=np.complex64)  # [time, angle] — complex for FFT

# --- Generate time-domain beamformed data (complex) ---
for i in range(T):
    label = random.randint(0, 1)
    labels.append(label)

    noise = (np.random.randn(M) + 1j * np.random.randn(M))
    noise /= np.linalg.norm(noise)

    if label == 1:
        f_shift = fc + np.random.uniform(-2, 2)
        snr = np.random.uniform(snr_min, 0)
        signal = np.exp(1j * 2 * np.pi * f_shift * d / c * np.cos(np.radians(doa)) * np.arange(M))
        signal /= np.linalg.norm(signal)
        amp = 10**(snr / 20)
        y = amp * signal + noise
    else:
        y = noise

    # Beamforming across angles
    S = np.exp(1j * 2 * np.pi * fc * d / c * np.cos(np.radians(Theta))[:, None] * np.arange(M))  # [angle, M]
    B = np.dot(S, y.conj())  # shape: (len(Theta),)
    Z[i, :] = B  # Save complex beamformed data for FFT

# --- FFT along time axis for each azimuth angle ---
Z_fft = np.fft.fft(Z, axis=0)  # shape: (T, angles)
freqs = np.fft.fftfreq(T, d=1/fs)  # Frequency axis

# --- Focus on bins around 0 Hz (we simulated narrowband signals)
k_center = np.abs(freqs - 0).argmin()
k_range = signal_band // 2
k_start = max(0, k_center - k_range)
k_end = min(T, k_center + k_range + 1)

# --- Extract magnitude spectrum and normalize per frame ---
Z_mag = np.abs(Z_fft[k_start:k_end, :])  # shape: (freq_bins, angles)
Z_norm = (Z_mag - np.mean(Z_mag)) / np.std(Z_mag)  # normalize full feature map

# --- Plot final normalized FFT beam domain as heatmap ---
plt.figure(figsize=(10, 5))
extent = [0, 180, freqs[k_end-1], freqs[k_start]]  # frequency on y-axis
plt.imshow(Z_norm, aspect='auto', extent=extent, cmap='plasma')
plt.colorbar(label='Normalized FFT Magnitude')
plt.xlabel("Azimuth (degrees)")
plt.ylabel("Frequency (Hz) from carrier")
plt.title("Beam Domain FFT (Normalized, Matching Paper Method)")
plt.tight_layout()
plt.show()


plt.plot(Theta, np.abs(Z_fft[k_center, :]))
plt.title("1D Beamformed Spectrum at Center Frequency")
plt.xlabel("Azimuth (°)")
plt.ylabel("FFT Magnitude")
plt.grid(True)
plt.show()
