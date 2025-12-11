from enum import IntEnum
from typing import Callable, Dict
from PIL import Image
import numpy as np
import cv2
# =========================
# 1. Action definitions & PIL - CV2 conversion
# =========================

class VLAAction(IntEnum):
    DO_NOTHING = 0
    SMART_CROP = 1
    BG_REMOVE = 2
    DENOISE = 3
    WHITE_BALANCE = 4
    SHARPEN = 5
    LIGHT_GEN_REFINE = 6 


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL → OpenCV (RGB → BGR)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV → PIL (BGR → RGB)."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# =========================
# 2. Image-processing functions
# (simple placeholders—you can replace with yours)
# =========================

def action_do_nothing(img: Image.Image) -> Image.Image:
    """Return the original image unchanged."""
    return img

def action_smart_crop(img: Image.Image) -> Image.Image:
    """Simple center crop (placeholder)."""
    w, h = img.size
    crop_ratio = 0.2  # remove 20% border
    left = int(crop_ratio * w / 2)
    top = int(crop_ratio * h / 2)
    right = w - left
    bottom = h - top
    return img.crop((left, top, right, bottom))

def action_bg_remove(img: Image.Image) -> Image.Image:
    """
    Perform background removal using GrabCut.
    Works well for objects with clear foreground.
    """
    cv = pil_to_cv(img)
    mask = np.zeros(cv.shape[:2], np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # rectangle covering almost entire image
    h, w = mask.shape
    rect = (10, 10, w - 20, h - 20)

    cv2.grabCut(cv, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    # mask: 0,2 = background, 1,3 = foreground
    fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")

    result = cv2.bitwise_and(cv, cv, mask=fg_mask)
    return cv_to_pil(result)

def action_denoise(img: Image.Image) -> Image.Image:
    """Denoise the image using FastNLMeans denoising."""
    cv = pil_to_cv(img)
    denoised = cv2.fastNlMeansDenoisingColored(
        cv, None,
        h=10,            # luminance filter strength
        hColor=10,       # chroma filter strength
        templateWindowSize=7,
        searchWindowSize=21
    )
    return cv_to_pil(denoised)

def action_white_balance(img: Image.Image) -> Image.Image:
    """
    Perform white balance using OpenCV's xphoto module.
    Works well for images with uneven lighting.
    """
    cv = pil_to_cv(img).astype(np.float32)

    # Compute mean per channel
    mean_b = cv[:, :, 0].mean()
    mean_g = cv[:, :, 1].mean()
    mean_r = cv[:, :, 2].mean()

    gray = (mean_b + mean_g + mean_r) / 2.0 #small number will add exposure

    # Compute scaling factors
    scale_b = gray / (mean_b + 1e-6)
    scale_g = gray / (mean_g + 1e-6)
    scale_r = gray / (mean_r + 1e-6)

    # Apply gains
    cv[:, :, 0] *= scale_b
    cv[:, :, 1] *= scale_g
    cv[:, :, 2] *= scale_r

    cv = np.clip(cv, 0, 255).astype(np.uint8)

    return cv_to_pil(cv)

def action_sharpen(img: Image.Image) -> Image.Image:
    """Sharpen the image using Gaussian blur and weighted addition."""
    cv = pil_to_cv(img)
    blurred = cv2.GaussianBlur(cv, (5, 5), sigmaX=1.0)
    sharpened = cv2.addWeighted(cv, 1.5, blurred, -0.5, 0)
    return cv_to_pil(sharpened)

def action_light_gen_refine(img: Image.Image) -> Image.Image:
    """
    Light generative refine = sharpen + denoise blend.
    Non-destructive "enhance" effect.
    """
    cv = pil_to_cv(img)

    # Denose
    den = cv2.fastNlMeansDenoisingColored(cv, None, 5, 5, 7, 21)

    # Light sharpen
    blur = cv2.GaussianBlur(den, (3, 3), 0)
    sharp = cv2.addWeighted(den, 1.3, blur, -0.3, 0)

    return cv_to_pil(sharp)


ACTION_FUNCS: Dict[VLAAction, Callable[[Image.Image], Image.Image]] = {
    VLAAction.DO_NOTHING: action_do_nothing,
    VLAAction.SMART_CROP: action_smart_crop,
    VLAAction.BG_REMOVE: action_bg_remove,
    VLAAction.DENOISE: action_denoise,
    VLAAction.WHITE_BALANCE: action_white_balance,
    VLAAction.SHARPEN: action_sharpen,
    VLAAction.LIGHT_GEN_REFINE: action_light_gen_refine,
}