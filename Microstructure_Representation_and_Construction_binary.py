#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# 设置整体字体风格适合论文
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
})

def generate_field_regularized(F_white, dx, alpha, sigma=2.0, k0=0.1, C=1.0, target_std=1.0):
    N = F_white.shape[0]
    kx = np.fft.fftfreq(N, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    exponent = alpha * np.log(K + k0) - (K**2) / (2*sigma**2)
    S = np.exp(exponent)
    M = np.max(S)
    S_norm = (S / M) * C if M > 0 else S

    A = np.sqrt(S_norm)
    F_field = A * F_white
    field2d = np.fft.ifftn(F_field).real
    field2d -= np.mean(field2d)
    std_val = np.std(field2d)
    if std_val > 0:
        field2d /= std_val
        field2d *= target_std
    return field2d, S_norm

def compute_power_spectrum_2d(field2d):
    F = np.fft.fftn(field2d)
    PS = np.abs(F)**2
    return np.fft.fftshift(PS)

if __name__ == "__main__":
    N = 512
    L = 50.0
    dx = L / N
    sigma = 2.0
    k0 = 0.1
    target_std = 1.0

    out_folder = "output_images"
    os.makedirs(out_folder, exist_ok=True)

    np.random.seed(42)
    white_noise = np.random.normal(0, 1, (N, N))
    F_white = np.fft.fftn(white_noise)

    alpha_list = [0, -2, -5, -10, -20, -50]

    n_rows = 3
    n_cols = len(alpha_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows),
                             gridspec_kw={'hspace': 0.01, 'wspace': 0.9})  # 控制紧凑度

    for j, alpha in enumerate(alpha_list):
        field, _ = generate_field_regularized(F_white, dx, alpha, sigma, k0=k0,
                                              C=1.0, target_std=target_std)
        PS = compute_power_spectrum_2d(field)
        binary_field = (field > 0).astype(float)

        # Top row: Log Power Spectrum
        ax_top = axes[0, j]
        im_top = ax_top.imshow(np.log10(PS + 1e-12), origin='lower', cmap='jet')
        ax_top.set_title(r"$\alpha = $" + f"{alpha}")
        ax_top.set_xlabel(r"$k_x$")
        ax_top.set_ylabel(r"$k_y$")
        fig.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.02)

        # Middle row: Continuous Field
        ax_mid = axes[1, j]
        field_crop = field[0:300, 0:300]
        im_mid = ax_mid.imshow(field_crop, origin='lower', cmap='jet')
        ax_mid.set_xlabel("x")
        ax_mid.set_ylabel("y")
        fig.colorbar(im_mid, ax=ax_mid, fraction=0.046, pad=0.02)

        # Bottom row: Binary Field
        ax_bot = axes[2, j]
        im_bot = ax_bot.imshow(binary_field, origin='lower', cmap='gray')
        ax_bot.set_xlabel("x")
        ax_bot.set_ylabel("y")
        fig.colorbar(im_bot, ax=ax_bot, fraction=0.046, pad=0.02)

    out_file = os.path.join(out_folder, "composite_field_cleaned_binary2.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')  # 高清输出
    plt.show()
    print(f"Saved compact figure to: {out_file}")