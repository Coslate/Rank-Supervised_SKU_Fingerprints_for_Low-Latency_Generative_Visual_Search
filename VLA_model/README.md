This repository provides a lightweight, modular set of **Vision–Language–Action (VLA)** image-processing actions intended for preprocessing, refinement, and enhancement tasks.

It includes:

- A set of image actions (crop, denoise, sharpen, background removal, etc.)
- A clean dispatch table (`ACTION_FUNCS`) indexed by `VLAAction` enums
- A test CLI script using `argparse` to run actions on any input image

Perfect for:
- Data preprocessing pipelines  
- VLA model input conditioning  
- Lightweight enhancement modules  
- Unit tests for vision components  

---


### Implemented Actions

| Action | Description |
|--------|-------------|
| **DO_NOTHING** | Returns original image |
| **SMART_CROP** | Center crop (removes border region) |
| **BG_REMOVE** | Background removal using GrabCut |
| **DENOISE** | `fastNlMeansDenoisingColored` for noise reduction |
| **WHITE_BALANCE** | Gray-world white balance (OpenCV + NumPy, platform-safe) |
| **SHARPEN** | Sharpen masking using Gaussian blur |
| **LIGHT_GEN_REFINE** | Light generative enhancement: denoise + sharpen blend |

All functions follow the signature:

```python
Callable[[Image.Image], Image.Image]
```

To test the function, use:
```bash
pip install -r requirements.txt
python test_image.py --input input.jpg --action ChooseYourAction --output out.jpg
# Example: python test_image.py --input sample.jpg --action SHARPEN --output out_sharp.jpg
```