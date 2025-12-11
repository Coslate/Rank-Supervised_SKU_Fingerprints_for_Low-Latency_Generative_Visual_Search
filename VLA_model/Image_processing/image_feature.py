import numpy as np
from PIL import Image
import cv2


def compute_quality_features(img: Image.Image):
    # Convert once to uint8 RGB
    rgb_u8 = np.array(img.convert("RGB"))          # uint8 in [0, 255]
    arr_rgb = rgb_u8.astype(np.float32) / 255.0    # float32 in [0, 1]

    # Grayscale: keep both uint8 and float32 versions
    gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)          # uint8
    arr_gray = gray_u8.astype(np.float32) / 255.0               # float32 in [0, 1]

    # --- 1. Luminance statistics ---
    lum = arr_rgb.mean(axis=2)
    lum_mean = lum.mean()
    lum_std = lum.std()
    lum_coef = lum_std / (lum_mean + 1e-6)

    # --- 2. Sharpness (Laplacian variance) ---
    # Use uint8 as input to Laplacian to avoid unsupported type combos
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = lap.var()

    # --- 3. Edge density ---
    edges = cv2.Canny(gray_u8, 100, 200)
    edge_density = edges.mean()

    # --- 4. Saturation statistics (HSV) ---
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    sat_mean = sat.mean()
    sat_std = sat.std()

    # --- 5. Noise level estimation ---
    # Estimate noise as high-frequency energy after a light blur
    blurred = cv2.GaussianBlur(arr_gray, (3, 3), 0)
    noise_map = arr_gray - blurred
    noise_level = np.mean(np.abs(noise_map))

    # --- 6. White balance deviation ---
    r_mean, g_mean, b_mean = arr_rgb.mean(axis=(0, 1))
    wb_dev = np.sqrt(
        (r_mean - g_mean) ** 2 +
        (g_mean - b_mean) ** 2 +
        (b_mean - r_mean) ** 2
    )

    # --- 7. Color variance ---
    color_variance = arr_rgb.reshape(-1, 3).var(axis=0).mean()

    return np.array([
        lum_mean, lum_std, lum_coef,
        sharpness, edge_density,
        sat_mean, sat_std,
        noise_level,
        wb_dev,
        color_variance
    ], dtype=np.float32)
