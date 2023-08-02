import base64
import io
import json
from io import BytesIO
from typing import Dict

from PIL import PngImagePlugin, Image
import piexif
import piexif.helper
import gradio as gr

from modules import scripts, script_callbacks, extensions, shared, ui
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.call_queue import queue_lock
import webui


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes,
                       format="PNG",
                       pnginfo=(metadata if use_metadata else None),
                       quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment:
                    piexif.helper.UserComment.dump(parameters or "",
                                                   encoding="unicode")
                }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes,
                           format="JPEG",
                           exif=exif_bytes,
                           quality=opts.jpeg_quality)
            else:
                image.save(output_bytes,
                           format="WEBP",
                           exif=exif_bytes,
                           quality=opts.jpeg_quality)

        else:
            # TODO
            print("Invalid image format")
        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def get_default_args(script_runner):
    #find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None] * last_arg_index
    script_args[0] = 0

    # get default values
    with gr.Blocks():  # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                script_args[script.args_from:script.
                            args_to] = ui_default_values
    return script_args


def simple_txt2img(args: Dict):
    webui.initialize()
    script_callbacks.before_ui_callback()
    shared.demo = ui.create_ui()

    ext_name = [iter.name for iter in extensions.extensions]
    print("[tangjicheng] extentions: ", ext_name)

    shared.refresh_checkpoints()

    model_name = args.pop("sd_model_checkpoint")
    shared.opts.set("sd_model_checkpoint", model_name)

    script_runner = scripts.scripts_txt2img
    if not script_runner.scripts:
        script_runner.initialize_scripts(False)

    # default_script_args = [0, 'NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\nINS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\nIND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\nINALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\nMIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\nOUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\nOUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\nOUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\nALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5', True, 'Disable', 'values', '0,0.25,0.5,0.75,1', 'Block ID', 'IN05-OUT05', 'none', '', '0.5,1', 'BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11', 1.0, 'black', '20', False, 'ATTNDEEPON:IN05-OUT05:attn:1\n\nATTNDEEPOFF:IN05-OUT05:attn:0\n\nPROJDEEPOFF:IN05-OUT05:proj:0\n\nXYZ:::1', False, False, False, 'positive', 'comma', 0, False, False, '', 'Seed', '', None, 'Nothing', '', None, 'Nothing', '', None, True, False, False, False, 0]

    default_script_args = get_default_args(script_runner)
    print(f"[tangjicheng] default_script_args: {default_script_args}")

    script_args = default_script_args

    args.pop('script_name', None)

    args.pop('script_args', None)
    args.pop('alwayson_scripts', None)
    send_images = args.pop('send_images', True)
    args.pop('save_images', None)

    with queue_lock:
        p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
        p.scripts = script_runner
        p.outpath_grids = opts.outdir_txt2img_grids
        p.outpath_samples = opts.outdir_txt2img_samples

        shared.state.begin()
        p.script_args = tuple(script_args)  # Need to pass args as tuple here
        processed = process_images(p)
        shared.state.end()

    b64images = list(map(encode_pil_to_base64,
                         processed.images)) if send_images else []
    return b64images


def simple_txt2img_test(input:str):
    print("[tangjicheng] Start simple_txt2img_test ...")

    model_input = json.loads(input)

    output = simple_txt2img(model_input)

    image = output[0]
    print(type(image))
    return image

def test_txt2img():
    input2 = '''{
                    "sd_model_checkpoint": "v1-5-pruned-emaonly",
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
                    "prompt": "1girl",
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

    image = simple_txt2img_test(input2)

    pic = Image.open(BytesIO(base64.b64decode(image)))
    pic.save("15.jpg")


# 1girl,  <lora:test_lora:1:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0>

if __name__ == "__main__":
    test_txt2img()
