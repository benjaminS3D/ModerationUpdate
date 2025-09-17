from PIL import Image
import base64
import io

def decode_data_url_to_image(data_url: str) -> Image.Image:
    if not data_url.startswith("data:"):
        raise ValueError("Expected data URL (data:*;base64,...)")
    try:
        b64 = data_url.split(",", 1)[1]
    except Exception:
        raise ValueError("Malformed data URL; missing comma separator.")
    img_bytes = base64.b64decode(b64, validate=False)
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")


def maybe_downscale(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)
