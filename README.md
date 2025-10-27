Post-Quantum Cryptography for Secure Image Transmission in 5G Networks

This project implements a secure image transmission system using Post-Quantum Cryptography (PQC) to ensure confidentiality and robustness against quantum attacks.
The system employs the Kyber algorithm, a lattice-based PQC scheme, for encrypting and decrypting image data transmitted over 5G networks.

---

Overview:

⦁	With the rise of quantum computing, traditional encryption methods like RSA and ECC are becoming vulnerable.
⦁	This project addresses that challenge by integrating Kyber (CRYSTALS-Kyber) — one of the NIST-selected PQC algorithms — to secure image data transmission in 5G environments.
⦁	The project performs encryption, transmission simulation, and decryption of images, followed by an analysis of image quality and encryption strength using various performance metrics.

---

Features:
⦁	Kyber-based Post-Quantum Encryption and Decryption
⦁	Image Encryption and Reconstruction
⦁	Simulation of 5G Transmission
⦁	Performance Evaluation using PSNR, MSE, and SSIM
⦁	Python-based implementation with clear modular design
⦁	Dataset support for multiple input images

---

Performance Metrics:

|           Metric                       |                          Description                                           |
|:---------------------------------------|:-------------------------------------------------------------------------------|
|  MSE (Mean Squared Error)              | Measures average squared difference between original and reconstructed images. |
|  PSNR (Peak Signal-to-Noise Ratio)     | Evaluates reconstruction quality after encryption/decryption.                  |
|  SSIM (Structural Similarity Index)    | Assesses perceptual similarity between images.                                 |

---

Requirements:
    - Python 3.10+
    - Required Libraries:
    ```bash
    pip install numpy opencv-python matplotlib scikit-image tqdm

How to Run:

⦁	    Clone the repository:
git clone https://github.com/your-username/Post-Quantum-Cryptography-for-Secure-Image-Transmission-in-5G-Networks.git
cd Post-Quantum-Cryptography-for-Secure-Image-Transmission-in-5G-Networks

⦁	    Run the main script:
         python main.py

The program will:
⦁	    Encrypt the image using Kyber PQC
⦁	    Simulate transmission
⦁	    Decrypt and reconstruct the image
⦁	    Display metrics: MSE, PSNR, SSIM

Dataset:
    The dataset/ folder contains the sample images used for encryption and decryption.
    You can replace these with your own images for testing.

Contributors:
    Shreya Saraswati
    Bhoomika B V
    Deeksha P T
    Yashodha B T
    Department of Electronics and Communication Engineering
    GSKSJTI, Bengaluru
    Major Project – Phase 2 (2025) (7th sem, VTU)

References:
⦁	    CRYSTALS-Kyber: https://pq-crystals.org/kyber/
⦁	    NIST PQC Standardization Project: https://csrc.nist.gov/projects/post-quantum-cryptography
⦁	    Research Papers on PQC-based Secure Communication in 5G Networks

This project demonstrates how quantum-safe cryptographic techniques can be integrated into real-world image transmission systems to ensure future-proof data security in next-generation wireless networks.