import numpy as np
import matplotlib.pyplot as plt
import os
import random
import zipfile
from scipy.fft import fft

# Function: Convert 32-point complex freq to 64-point real time signal
def complex32_to_real64_ifft(freq_32_input):
    freq_32 = freq_32_input.copy()
    freq_32[0] = np.real(freq_32[0])       # DC
    freq_32[-1] = np.real(freq_32[-1])     # Nyquist
    freq_64 = np.zeros(64, dtype=np.complex64)
    freq_64[:32] = freq_32
    freq_64[33:] = np.conj(freq_32[1:][::-1])  # Correct Hermitian symmetry
    time_64 = np.fft.ifft(freq_64)
    return np.real(time_64)

# Parameters
T = 100
c = 1500
f = 1000
M = 32
d = 0.45 * c / f
Theta = np.arange(0, 181)
num_simulations = 100
start_index = 0

# Folders
base_dir = "simulation_resultsD"
folders = {
    "z_intermediate": os.path.join(base_dir, "Z_Intermediate_NPY"),
    "z_final": os.path.join(base_dir, "Z_Final_NPY"),
    "y_input": os.path.join(base_dir, "Y_Input_IFFT64"),
    "ground_npy": os.path.join(base_dir, "Ground_Truth_NPY"),
    "btr_img": os.path.join(base_dir, "BTR_Images"),
    "ground_img": os.path.join(base_dir, "Ground_Truth_Images")
}
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# Simulate
for i in range(num_simulations):
    sim_number = start_index + i
    signal_type = 'const_doa'
    snr = random.randint(0, 20)
    doa_start = np.random.randint(5, 175)

    Z = np.zeros((T, len(Theta)))          # Z_intermediate
    Z_gt = np.zeros_like(Z)                # Ground truth
    Y_input = np.zeros((T, 64))            # IFFT signal
    doa_vals = np.zeros(T)

    for t in range(T):
        doa = np.clip(doa_start, 0, 180)
        doa_vals[t] = doa

        y = np.exp(-1j * 2 * np.pi * f * d / c * np.cos(np.radians(doa)) * np.arange(M))
        noise = np.random.randn(M)
        y = 10 ** (snr / 20) * y + noise

        freq_32 = fft(y)
        time_64_real = complex32_to_real64_ifft(freq_32)
        Y_input[t, :] = time_64_real

        S = np.exp(1j * 2 * np.pi * f * d / c * np.cos(np.radians(Theta))[:, None] * np.arange(M))
        B = np.dot(y, S.T)
        power = np.abs(B) ** 2
        P = 10 * np.log10(power + 1e-12)
        Z[t, :] = P

        doa_idx = int(np.clip(np.round(doa), 0, 180))
        Z_gt[t, doa_idx] = 1

    # Final Z: Expand to (100, 181, 64)
    Z_final = np.zeros((T, 181, 64))  # Shape: (time, DOA, time-domain signal length)

    for t in range(T):
        # For each time step, apply the same time-domain signal Y_input[t]
        # across all 181 DOA bins (broadcast)
        Z_final[t, :, :] = np.tile(Y_input[t], (181, 1))


    # Save files
    np.save(os.path.join(folders["z_intermediate"], f"Z_intermediate_SNR_{snr}_{sim_number}.npy"), Z)
    np.save(os.path.join(folders["z_final"], f"Z_final_SNR_{snr}_{sim_number}.npy"), Z_final)
    np.save(os.path.join(folders["y_input"], f"Y_input_IFFT64_SNR_{snr}_{sim_number}.npy"), Y_input)
    np.save(os.path.join(folders["ground_npy"], f"Ground_Truth_SNR_P_{snr}_{sim_number}.npy"), Z_gt)

    # Images for Z
    plt.figure()
    plt.pcolor(Theta, np.arange(T), Z, shading='auto')
    plt.xlabel('Bearing')
    plt.ylabel('Time')
    plt.title(f"BTR_Simulation_SNR_P_{snr}_{sim_number}")
    plt.savefig(os.path.join(folders["btr_img"], f"BTR_Simulation_SNR_P_{snr}_{sim_number}.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # Image for Z_gt
    plt.figure()
    plt.imshow(Z_gt, aspect='auto', cmap='viridis', origin='lower')
    plt.xlabel('Bearing')
    plt.ylabel('Time')
    plt.title(f"Ground_Truth_SNR_P_{snr}_{sim_number}")
    plt.savefig(os.path.join(folders["ground_img"], f"Ground_Truth_SNR_P_{snr}_{sim_number}.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

# Zip helper
def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

# Zip everything
for name, path in folders.items():
    zip_folder(path, os.path.join(base_dir, f"{os.path.basename(path)}.zip"))

print("✅ Simulation complete — Z_intermediate and Z_final saved separately and zipped.")
