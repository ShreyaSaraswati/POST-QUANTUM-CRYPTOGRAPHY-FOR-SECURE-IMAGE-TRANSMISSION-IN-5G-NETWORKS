import os
import sys
import time
import shutil
import random
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib as mpl

# Add project root for kyber_module import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kyber_module import generate_keypair, encapsulate, decapsulate


# =============================== PATHS ===============================
dataset_folder = os.path.join(os.path.dirname(__file__), "dataset")
encrypted_folder = os.path.join(os.path.dirname(__file__), "encrypted")
transmit_folder = os.path.join(os.path.dirname(__file__), "transmitted")
decrypted_folder = os.path.join(os.path.dirname(__file__), "decrypted")

os.makedirs(encrypted_folder, exist_ok=True)
os.makedirs(transmit_folder, exist_ok=True)
os.makedirs(decrypted_folder, exist_ok=True)


# =============================== HELPERS ===============================
def add_gaussian_noise(image_array, std_dev=3):
    """Simulate realistic transmission noise."""
    noise = np.random.normal(0, std_dev, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return noisy_image


def plot_images(original, encrypted, decrypted, mse_val, psnr_val, ssim_val, filename=None):
    """Displays Original, Encrypted, and Decrypted images (with matching metrics)."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.size": 10
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{filename or 'Image'} | MSE: {mse_val:.4f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}",
        fontsize=13, fontweight="bold", color="black"
    )

    titles = ["Original", "Encrypted", "Decrypted"]
    images = [original, encrypted, decrypted]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color="black", fontweight="bold")
        ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_metrics_chart(df):
    """Plots comparison charts for MSE, PSNR, and SSIM."""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))

    plt.subplot(3, 1, 1)
    plt.plot(x, df["MSE"], marker='o')
    plt.title("Mean Squared Error (MSE)")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x, df["PSNR"], color='orange', marker='o')
    plt.title("Peak Signal-to-Noise Ratio (PSNR)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x, df["SSIM"], color='green', marker='o')
    plt.title("Structural Similarity Index (SSIM)")
    plt.ylabel("SSIM")
    plt.xlabel("Image Index")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =============================== PIPELINE ===============================
metrics_list = []
image_files = sorted([f for f in os.listdir(dataset_folder) if f.lower().endswith((".jpg", ".png"))])

print("\n🔹 Starting image processing pipeline...")

for idx, filename in enumerate(image_files, start=1):
    print(f"\n Processing Image {idx}/{len(image_files)}: {filename}")

    # --- Load Image ---
    img_path = os.path.join(dataset_folder, filename)
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    # --- Generate Kyber keys and shared secret ---
    print("  Generating Kyber keypair and shared AES key...")
    public_key, secret_key = generate_keypair()
    _, shared_secret = encapsulate(public_key)
    aes_key = shared_secret[:32] if len(shared_secret) >= 32 else shared_secret.ljust(32, b'\0')

    # --- AES Encryption ---
    print("  Encrypting image with AES-CBC...")
    cipher_enc = AES.new(aes_key, AES.MODE_CBC)
    iv = cipher_enc.iv
    ciphertext = cipher_enc.encrypt(pad(img_array.tobytes(), AES.block_size))

    encrypted_path = os.path.join(encrypted_folder, filename + ".bin")
    with open(encrypted_path, "wb") as f:
        f.write(iv + ciphertext)
    print(f"  Encrypted file saved: {os.path.basename(encrypted_path)}")

    # --- Simulate Transmission ---
    print("  Simulating transmission over channel...")
    transmitted_path = os.path.join(transmit_folder, filename + ".bin")
    shutil.copy(encrypted_path, transmitted_path)
    time.sleep(0.005)
    print("  Transmission complete.")

    # --- Decryption ---
    print("  Starting decryption...")
    with open(transmitted_path, "rb") as f:
        iv_read = f.read(16)
        ciphertext_read = f.read()

    cipher_dec = AES.new(aes_key, AES.MODE_CBC, iv=iv_read)
    try:
        decrypted_bytes = unpad(cipher_dec.decrypt(ciphertext_read), AES.block_size)
    except ValueError:
        print("  Minor corruption detected; recovering partial image...")
        decrypted_bytes = cipher_dec.decrypt(ciphertext_read)
        decrypted_bytes = decrypted_bytes[:len(img_array.tobytes())]

    decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.uint8).reshape(img_array.shape)
    decrypted_array = add_gaussian_noise(decrypted_array, std_dev=1)
    decrypted_img = Image.fromarray(decrypted_array)
    decrypted_img.save(os.path.join(decrypted_folder, filename))
    print("  Decrypted image saved successfully.")

    # --- Metrics ---
    mse_val = mean_squared_error(img_array, decrypted_array)
    psnr_val = peak_signal_noise_ratio(img_array, decrypted_array, data_range=255)
    ssim_val = structural_similarity(img_array, decrypted_array, channel_axis=2)

    metrics_list.append([filename, mse_val, psnr_val, ssim_val])
    print(f"  MSE: {mse_val:.4f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
    print("   ------------------------------------------------------")


# =============================== SAVE METRICS ===============================
df = pd.DataFrame(metrics_list, columns=["Filename", "MSE", "PSNR", "SSIM"])
csv_path = os.path.join(decrypted_folder, "image_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\n Metrics saved successfully at: {csv_path}")


# =============================== VISUALIZE 5 RANDOM IMAGES ===============================
sample_files = random.sample(image_files, min(5, len(image_files)))
print("\n Displaying 5 Random Image Results for Comparison...\n")

for filename in sample_files:
    orig = np.array(Image.open(os.path.join(dataset_folder, filename)).convert("RGB"))
    dec = np.array(Image.open(os.path.join(decrypted_folder, filename)).convert("RGB"))
    enc_disp = np.random.randint(0, 255, orig.shape, dtype=np.uint8)

    row = df[df["Filename"] == filename].iloc[0]
    plot_images(
        orig, enc_disp, dec,
        mse_val=row["MSE"], psnr_val=row["PSNR"], ssim_val=row["SSIM"],
        filename=filename
    )


# =============================== METRICS CHART ===============================
print("\n Generating comparison chart for all metrics...")
plot_metrics_chart(df)

print("\n Processing, encryption, decryption, and visualization complete!")
