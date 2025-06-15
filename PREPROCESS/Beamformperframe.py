import numpy as np
import matplotlib.pyplot as plt
import random

# --- Parameters ---
T = 200                      # Number of frames (time windows)
M = 32                       # Hydrophones
fs = 1000                   # Sampling rate (Hz)
fc = 1000                   # Center frequency (Hz)
c = 1500                    # Speed of sound
d = 0.45 * c / fc           # Element spacing
Theta = np.arange(0, 181)   # 181 azimuth angles
D = len(Theta)              # 181 directions
signal_band = 5             # ±2 bins around 0 Hz
signal_len = 100            # Samples per frame (time points)
snr_min = -14               # SNR range
snr_max = 0

X = []
y = []

# --- Main Loop: T frames ---
for frame in range(T):
    doa = np.random.uniform(0, 180)  # Random target DOA for this frame
    label_angles = [int(round(doa))]  # Single-angle label for now

    # Generate multichannel signal
    signal_present = random.randint(0, 1)
    noise = (np.random.randn(signal_len, M) + 1j * np.random.randn(signal_len, M))

    if signal_present:
        # Create time signal
        t = np.arange(signal_len) / fs
        f_shift = fc + np.random.uniform(-2, 2)
        snr = np.random.uniform(snr_min, snr_max)

        # Steering vector for target direction
        sv = np.exp(-1j * 2 * np.pi * f_shift * d / c * np.cos(np.radians(doa)) * np.arange(M))
        tone = np.exp(1j * 2 * np.pi * f_shift * t).reshape(-1, 1)

        # Broadcast across hydrophones
        sig = tone * sv
        sig = np.real(sig)  # Passive sonar measures real pressure

        # Normalize and scale to SNR
        sig_power = np.mean(sig**2)
        noise_power = np.mean(noise.real**2)
        scale = np.sqrt(noise_power * 10**(snr / 10) / sig_power)
        sig *= scale

        x = sig + noise.real
    else:
        x = noise.real

    # --- Beamforming for each angle ---
    for i, angle in enumerate(Theta):
        steering = np.exp(-1j * 2 * np.pi * fc * d / c * np.cos(np.radians(angle)) * np.arange(M))
        beamformed = np.dot(x, steering.conj())  # shape: (signal_len,)

        # FFT
        B_fft = np.fft.fft(beamformed)
        freqs = np.fft.fftfreq(signal_len, d=1/fs)

        # Keep bins near 0 Hz
        k_center = np.abs(freqs - 0).argmin()
        k_range = signal_band // 2
        k_start = max(0, k_center - k_range)
        k_end = min(len(freqs), k_center + k_range + 1)

        spectrum = B_fft[k_start:k_end]
        feature = np.concatenate([np.real(spectrum), np.imag(spectrum)])

        # Label = 1 if this beam direction ≈ DOA
        is_target = int(i in label_angles and signal_present)
        X.append(feature)
        y.append(is_target)

# Final shape
X = np.array(X)
y = np.array(y)

print(f"X.shape = {X.shape}")  # Expect (T * 181, 2 * signal_band)
print(f"y.shape = {y.shape}")  # Expect (T * 181,)
