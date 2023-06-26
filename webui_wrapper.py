import webui
from modules.api import api

import base64
import io
import time
import datetime
import uvicorn
from threading import Lock
from io import BytesIO

import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, postprocessing
from modules.api import models
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin,Image
from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights, list_models
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import Dict, List, Any
import piexif
import piexif.helper

from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, queue_lock  # noqa: F401

from modules.api import api_log

def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        pass
        # TODO
        # raise 

    return name

def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            # TODO
            pass
            # raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        pass
        # TODO
        # raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e

class ApiWrapper:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]


    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def init_default_script_args(self, script_runner):
        #find max idx from the scripts in runner and generate a none array to init script_args
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # None everywhere except position 0 to initialize script args
        script_args = [None]*last_arg_index
        script_args[0] = 0

        # get default values
        # with gr.Blocks(): # will throw errors calling ui function without this
        #     for script in script_runner.scripts:
        #         if script.ui(script.is_img2img):
        #             ui_default_values = []
        #             for elem in script.ui(script.is_img2img):
        #                 ui_default_values.append(elem.value)
        #             script_args[script.args_from:script.args_to] = ui_default_values
        return script_args

    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts and (len(request.alwayson_scripts) > 0):
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script is None:
                    pass
                    # TODO
                    # raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson is False:
                    pass
                    # TODO
                    # raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # min between arg length in scriptrunner and arg length in the request
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
        return script_args


    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        print(f"{api_log.get_log_head()} called text2imgapi")
        list_models()
        shared.opts.set("sd_model_checkpoint", "Lyriel.safetensors")
        shared.opts.save(shared.config_filename)

        script_runner = scripts.scripts_txt2img
        # if not script_runner.scripts:
        #     script_runner.initialize_scripts(False)
        #     ui.create_ui()
        # if not self.default_script_arg_txt2img:
        #     self.default_script_arg_txt2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_txt2img_grids
            p.outpath_samples = opts.outdir_txt2img_samples

            shared.state.begin()
            if selectable_scripts is not None:
                p.script_args = script_args
                processed = scripts.scripts_txt2img.run(p, *p.script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

class StableDiffusionWebuiWrapper:
    def __init__(self):
        self.api = ApiWrapper(queue_lock=queue_lock)

    def initialize(self):
        webui.initialize()

    def txt2img(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI) -> models.TextToImageResponse:
        return self.api.text2imgapi(txt2imgreq)

if __name__ == "__main__":
    print("start testing for StableDiffusionWebuiWrapper ...")
    webui = StableDiffusionWebuiWrapper()
    # print(f"type: {type(models.StableDiffusionTxt2ImgProcessingAPI)}")
    input = models.StableDiffusionTxt2ImgProcessingAPI(
        enable_hr = False,
        denoising_strength = 0.5,
        # "firstphase_width": 0,
        # "firstphase_height": 0,
        # "hr_scale": 2,
        # "hr_upscaler": "string",
        # "hr_second_pass_steps": 0,
        # "hr_resize_x": 0,
        # "hr_resize_y": 0,
        # "hr_sampler_name": "string",
        # "hr_prompt": "",
        # "hr_negative_prompt": "",
        prompt = "1girl",
        # "styles": [
        #     "string"
        # ],
        # "seed": -1,
        # "subseed": -1,
        # "subseed_strength": 0,
        # "seed_resize_from_h": -1,
        # "seed_resize_from_w": -1,
        # "sampler_name": "LMS",
        # "batch_size": 1,
        # "n_iter": 1,
        # "steps": 20,
        # "cfg_scale": 7,
        # "width": 512,
        # "height": 512,
        # "restore_faces": false,
        # "tiling": false,
        # "do_not_save_samples": false,
        # "do_not_save_grid": false,
        # "negative_prompt": "",
        # "eta": 0,
        # "s_min_uncond": 0,
        # "s_churn": 0,
        # "s_tmax": 0,
        # "s_tmin": 0,
        # "s_noise": 1,
        # "override_settings": {},
        # "override_settings_restore_afterwards": true,
        # "script_args": [],
        # "sampler_index": "Euler",
        # "script_name": "",
        # "send_images": true,
        # "save_images": false,
        # "alwayson_scripts": {}
    )
    webui.txt2img(input)