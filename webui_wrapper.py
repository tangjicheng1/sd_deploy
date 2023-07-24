import base64
import io
import json
from io import BytesIO
from typing import Dict

from PIL import PngImagePlugin, Image
import piexif
import piexif.helper

import modules.shared as shared
from modules import scripts
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.call_queue import queue_lock


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(
                metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode")}
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG",
                           exif=exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP",
                           exif=exif_bytes, quality=opts.jpeg_quality)

        else:
            # TODO
            print("Invalid image format")
        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def simple_txt2img(args: Dict):
    shared.refresh_checkpoints()

    model_name = args.pop("sd_model_checkpoint")
    shared.opts.set("sd_model_checkpoint", model_name)

    script_runner = scripts.scripts_txt2img

    args.pop('script_name', None)

    args.pop('script_args', None)
    args.pop('alwayson_scripts', None)
    send_images = args.pop('send_images', True)
    args.pop('save_images', None)
    script_args = []

    with queue_lock:
        p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
        p.scripts = script_runner
        p.outpath_grids = opts.outdir_txt2img_grids
        p.outpath_samples = opts.outdir_txt2img_samples

        shared.state.begin()
        p.script_args = tuple(script_args)  # Need to pass args as tuple here
        processed = process_images(p)
        shared.state.end()

    b64images = list(map(encode_pil_to_base64, processed.images)
                     ) if send_images else []
    return b64images


def test_txt2img():
    print("[tangjicheng] Start testing text to image...")
    input2 = '''{
                    "sd_model_checkpoint": "Deliberate.safetensors",
                    "enable_hr": false,
                    "denoising_strength": 0.5,
                    "firstphase_width": 0,
                    "firstphase_height": 0,
                    "hr_scale": 2,
                    "hr_upscaler": "string",
                    "hr_second_pass_steps": 0,
                    "hr_resize_x": 0,
                    "hr_resize_y": 0,
                    "hr_sampler_name": "string",
                    "hr_prompt": "",
                    "hr_negative_prompt": "",
                    "prompt": "complex 3d render ultra detailed of a beautiful porcelain profile woman android face, cyborg, robotic parts, 150 mm, beautiful studio soft light, rim light, vibrant details, luxurious cyberpunk, lace, hyperrealistic, anatomical, facial muscles, cable electric wires, microchip, elegant, beautiful background, octane render, H. R. Giger style, 8k, best quality, masterpiece, illustration, an extremely delicate and beautiful, extremely detailed ,CG ,unity ,wallpaper, (realistic, photo-realistic:1.37),Amazing, finely detail, masterpiece,best quality,official art, extremely detailed CG unity 8k wallpaper, absurdres, incredibly absurdres, robot, silver halmet, full body, sitting, (masterpiece), (best quality:1.2), absurdres, (durex:1.3), condom box, condoms, durex classic jeans,8k, RAW photo, best quality, ultra high res, photorealistic, nude, full body, thigh, marie rose,",
                    "styles": [
                        "string"
                    ],
                    "seed": 123,
                    "subseed": 123,
                    "subseed_strength": 0,
                    "seed_resize_from_h": -1,
                    "seed_resize_from_w": -1,
                    "sampler_name": "LMS",
                    "batch_size": 1,
                    "n_iter": 1,
                    "steps": 20,
                    "cfg_scale": 7,
                    "width": 512,
                    "height": 512,
                    "restore_faces": false,
                    "tiling": false,
                    "do_not_save_samples": false,
                    "do_not_save_grid": false,
                    "negative_prompt": "",
                    "eta": 0,
                    "s_min_uncond": 0,
                    "s_churn": 0,
                    "s_tmax": 0,
                    "s_tmin": 0,
                    "s_noise": 1,
                    "override_settings": {},
                    "override_settings_restore_afterwards": true,
                    "script_args": [],
                    "sampler_index": "Euler",
                    "script_name": "",
                    "send_images": true,
                    "save_images": false,
                    "alwayson_scripts": {}
                    }'''

    model_input = json.loads(input2)

    output = simple_txt2img(model_input)

    image = output[0]

    pic = Image.open(BytesIO(base64.b64decode(image)))
    pic.save("3.jpg")


if __name__ == "__main__":
    test_txt2img()
