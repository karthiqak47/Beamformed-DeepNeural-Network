import numpy as np
import matplotlib.pyplot as plt
import random

# --- Parameters ---
T = 200                     # Number of frames (time steps)
c = 1500                    # Speed of sound (m/s)
fc = 1000                   # Center frequency (Hz)
fs = 4000                   # Sampling rate (Hz)
M = 64                      # Number of hydrophones
d = 0.5 * c / fc            # Half-wavelength spacing
Theta = np.arange(0, 181)   # Beam angles (0° to 180°)
snr_min = -14               # Minimum SNR (dB)

Ts = 256  # Samples per frame
labels = []
Z = np.zeros((T, Ts, len(Theta)), dtype=np.complex64)  # [frame, time, angle]

# --- Choose DOA for signal ---
doa = np.random.uniform(0, 180)
print(f"True DOA used in simulation: {doa:.2f}°")

# --- Steering matrix (for beamforming) ---
steering = np.exp(-1j * 2 * np.pi * fc * d / c *
                  np.cos(np.radians(Theta))[:, None] * np.arange(M))  # [angles, M]

# --- Loop over T frames ---
for t_idx in range(T):
    label = random.randint(0, 1)
    labels.append(label)

    # --- Generate noise
    noise = (np.random.randn(M, Ts) + 1j * np.random.randn(M, Ts)) / np.sqrt(2)

    if label == 1:
        f_shift = fc + np.random.uniform(-2, 2)
        snr = np.random.uniform(snr_min, 0)
        amp = 10 ** (snr / 20)
        t = np.arange(Ts) / fs
        tone = np.exp(1j * 2 * np.pi * f_shift * t)

        # Apply DOA-based delays
        delays = d * np.arange(M) * np.cos(np.radians(doa)) / c
        sample_delays = np.round(delays * fs).astype(int)

        signal = np.zeros((M, Ts), dtype=np.complex64)
        for m in range(M):
            shift = sample_delays[m]
            shifted = np.roll(tone, shift)
            shifted[:shift] = 0
            signal[m, :] = amp * shifted

        y = signal + noise
    else:
        y = noise

    # --- Beamform over all angles
    beam_output = np.dot(steering.conj(), y)  # [angles, Ts]
    Z[t_idx] = beam_output.T  # [time, angle]




# --- FFT over time across all frames ---
Z_fft = np.fft.fft(Z, axis=1)
freqs = np.fft.fftfreq(Ts, d=1/fs)
pos_idx = freqs > 0

Z_mag = np.abs(Z_fft[:, pos_idx, :])  # [frame, freq, angle]
Z_avg = np.mean(Z_mag, axis=0)        # [freq, angle]
Z_norm = (Z_avg - np.mean(Z_avg)) / np.std(Z_avg)

# --- Plot heatmap ---
plt.figure(figsize=(10, 5))
extent = [0, 180, freqs[pos_idx][-1], freqs[pos_idx][0]]
plt.imshow(Z_norm, aspect='auto', extent=extent, cmap='plasma')
plt.axvline(doa, color='cyan', linestyle='--', label=f"True DOA = {doa:.1f}°")
plt.colorbar(label='Normalized FFT Magnitude')
plt.xlabel("Azimuth (degrees)")
plt.ylabel("Frequency (Hz)")
plt.title("Beamformed Time-Frequency Spectrum")
plt.legend() 
plt.tight_layout()
plt.show()

# --- Plot 1D spectrum at 1000 Hz bin ---
k_center = np.abs(freqs - fc).argmin()
plt.figure(figsize=(10, 4))
plt.plot(Theta, np.abs(Z_fft[:, k_center, :]).mean(axis=0))
plt.axvline(doa, color='red', linestyle='--', label=f"True DOA = {int(round(doa))}°")
plt.title(f"1D Beamformed Spectrum at {fc} Hz")
plt.xlabel("Azimuth (°)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
extent = [0, 180, freqs[pos_idx][-1], freqs[pos_idx][0]]
plt.imshow(Z_norm, aspect='auto', extent=extent, cmap='plasma')
plt.axvline(doa, color='cyan', linestyle='--', linewidth=2, label=f"True DOA = {doa:.1f}°")
plt.scatter([doa], [fc], color='white', s=60, edgecolors='black', zorder=10, label="Signal Peak")
plt.colorbar(label='Normalized FFT Magnitude')
plt.xlabel("Azimuth (degrees)")
plt.ylabel("Frequency (Hz)")
plt.title("Beamformed Time-Frequency Spectrum (with Ground Truth)")
plt.legend()
plt.tight_layout()
plt.show()



# --- Plot 1D Beamformed Spectrum at fc ---
k_center = np.abs(freqs - fc).argmin()
beam_spectrum_fc = np.abs(Z_fft[:, k_center, :])  # [frame, angle]

plt.figure(figsize=(10, 5))
plt.imshow(beam_spectrum_fc.T, aspect='auto', cmap='hot', origin='lower',
           extent=[0, T, 0, 180])
plt.axhline(y=doa, color='cyan', linestyle='--', label=f"True DOA = {doa:.1f}°")

# Overlay label as bar at bottom
labels_array = np.array(labels) * 10  # scale for plotting
plt.plot(np.arange(T), labels_array, color='lime', lw=1.5, label="Signal Present")

plt.xlabel("Frame Index")
plt.ylabel("DOA (°)")
plt.title(f"DOA Detection over Time at {fc} Hz")
plt.legend()
plt.tight_layout()
plt.show()

print()
