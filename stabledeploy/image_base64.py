import io
import base64
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper

def decode_base64_to_image(encoding):
    image = Image.open(io.BytesIO(base64.b64decode(encoding)))
    return image


def encode_pil_to_base64(image, image_format="jpeg", jpeg_quality=90):
    with io.BytesIO() as output_bytes:

        if image_format == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=jpeg_quality)

        elif image_format in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if image_format in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=jpeg_quality)

        else:
            print("[encode error] Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)
