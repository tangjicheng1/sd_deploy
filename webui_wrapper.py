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


def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except Exception as e:
        # TODO
        print("Error: upscaler_to_index")
        # raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in shared.sd_upscalers])}") from e


def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        print(f"Script '{name}' not found")
        # raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        print("Sampler not found")
        # raise HTTPException(status_code=404, detail="Sampler not found")

    return name


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        # TODO
        print("Error: decode_base64_to_image, Invalid encoded image")
        # raise HTTPException(status_code=500, detail="Invalid encoded image") from e


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
            print("Invalid image format")
            # raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

class ApiWrapper:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None
        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]

        return models.ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    def get_script_info(self):
        res = []

        for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts]:
            res += [script.api_info for script in script_list if script.api_info is not None]

        return res

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

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
                    print(f"always on script {alwayson_script_name} not found")
                    # TODO
                    # raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson is False:
                    print("Cannot have a selectable script in the always on scripts params")
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
        script_runner = scripts.scripts_txt2img
        # if not script_runner.scripts:
        #     script_runner.initialize_scripts(False)
        #     # ui.create_ui()
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

    def img2imgapi(self, img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
        print(f"{api_log.get_log_head()} called img2imgapi")
        init_images = img2imgreq.init_images
        if init_images is None:
            # TODO
            print("Init image not found")
            # raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            # ui.create_ui()
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(img2imgreq.script_name, script_runner)

        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop('script_name', None)
        args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
            p.init_images = [decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = opts.outdir_img2img_grids
            p.outpath_samples = opts.outdir_img2img_samples

            shared.state.begin()
            if selectable_scripts is not None:
                p.script_args = script_args
                processed = scripts.scripts_img2img.run(p, *p.script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return models.ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: models.ExtrasSingleImageRequest):
        print(f"{api_log.get_log_head()} called extras_single_image_api")
        reqDict = setUpscalers(req)

        reqDict['image'] = decode_base64_to_image(reqDict['image'])

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: models.ExtrasBatchImagesRequest):
        print(f"{api_log.get_log_head()} called extras_batch_images_api")
        reqDict = setUpscalers(req)

        image_list = reqDict.pop('imageList', [])
        image_folder = [decode_base64_to_image(x.data) for x in image_list]

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasBatchImagesResponse(images=list(map(encode_pil_to_base64, result[0])), html_info=result[1])

    def pnginfoapi(self, req: models.PNGInfoRequest):
        print(f"{api_log.get_log_head()} called pnginfoapi")
        if(not req.image.strip()):
            return models.PNGInfoResponse(info="")

        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        items = {**{'parameters': geninfo}, **items}

        return models.PNGInfoResponse(info=geninfo, items=items)

    def progressapi(self, req: models.ProgressRequest):
        print(f"{api_log.get_log_head()} called progressapi")
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    def interrogateapi(self, interrogatereq: models.InterrogateRequest):
        print(f"{api_log.get_log_head()} called interrogateapi")
        image_b64 = interrogatereq.image
        if image_b64 is None:
            print("Error: interrogateapi, Image not found")
            # raise HTTPException(status_code=404, detail="Image not found")

        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                print("Error: interrogateapi, Model not found")
                # raise HTTPException(status_code=404, detail="Model not found")

        return models.InterrogateResponse(caption=processed)

    def interruptapi(self):
        print(f"{api_log.get_log_head()} called interruptapi")
        shared.state.interrupt()

        return {}

    def unloadapi(self):
        print(f"{api_log.get_log_head()} called unloadapi")
        unload_model_weights()

        return {}

    def reloadapi(self):
        print(f"{api_log.get_log_head()} called reloadapi")
        reload_model_weights()

        return {}

    def skip(self):
        print(f"{api_log.get_log_head()} called skip")
        shared.state.skip()

    def get_config(self):
        print(f"{api_log.get_log_head()} called get_config")
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if(metadata is not None):
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def set_config(self, req: Dict[str, Any]):
        print(f"{api_log.get_log_head()} called set_config")
        for k, v in req.items():
            shared.opts.set(k, v)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        print(f"{api_log.get_log_head()} called get_cmd_flags")
        return vars(shared.cmd_opts)

    def get_samplers(self):
        print(f"{api_log.get_log_head()} called get_samplers")
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_upscalers(self):
        print(f"{api_log.get_log_head()} called get_upscalers")
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    def get_sd_models(self):
        print(f"{api_log.get_log_head()} called get_sd_models")
        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": find_checkpoint_config_near_filename(x)} for x in checkpoints_list.values()]

    def get_hypernetworks(self):
        print(f"{api_log.get_log_head()} called get_hypernetworks")
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        print(f"{api_log.get_log_head()} called get_face_restorers")
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_realesrgan_models(self):
        print(f"{api_log.get_log_head()} called get_realesrgan_models")
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    def get_prompt_styles(self):
        print(f"{api_log.get_log_head()} called get_prompt_styles")
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_embeddings(self):
        print(f"{api_log.get_log_head()} called get_embeddings")
        db = sd_hijack.model_hijack.embedding_db

        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        return {
            "loaded": convert_embeddings(db.word_embeddings),
            "skipped": convert_embeddings(db.skipped_embeddings),
        }

    def refresh_checkpoints(self):
        print(f"{api_log.get_log_head()} called refresh_checkpoints")
        shared.refresh_checkpoints()

    def create_embedding(self, args: dict):
        print(f"{api_log.get_log_head()} called create_embedding")
        try:
            shared.state.begin()
            filename = create_embedding(**args) # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
            shared.state.end()
            return models.CreateResponse(info=f"create embedding filename: {filename}")
        except AssertionError as e:
            shared.state.end()
            return models.TrainResponse(info=f"create embedding error: {e}")

    def create_hypernetwork(self, args: dict):
        print(f"{api_log.get_log_head()} called create_hypernetwork")
        try:
            shared.state.begin()
            filename = create_hypernetwork(**args) # create empty embedding
            shared.state.end()
            return models.CreateResponse(info=f"create hypernetwork filename: {filename}")
        except AssertionError as e:
            shared.state.end()
            return models.TrainResponse(info=f"create hypernetwork error: {e}")

    def preprocess(self, args: dict):
        print(f"{api_log.get_log_head()} called preprocess")
        try:
            shared.state.begin()
            preprocess(**args) # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return models.PreprocessResponse(info = 'preprocess complete')
        except KeyError as e:
            shared.state.end()
            return models.PreprocessResponse(info=f"preprocess error: invalid token: {e}")
        except AssertionError as e:
            shared.state.end()
            return models.PreprocessResponse(info=f"preprocess error: {e}")
        except FileNotFoundError as e:
            shared.state.end()
            return models.PreprocessResponse(info=f'preprocess error: {e}')

    def train_embedding(self, args: dict):
        print(f"{api_log.get_log_head()} called train_embedding")
        try:
            shared.state.begin()
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                embedding, filename = train_embedding(**args) # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except AssertionError as msg:
            shared.state.end()
            return models.TrainResponse(info=f"train embedding error: {msg}")

    def train_hypernetwork(self, args: dict):
        print(f"{api_log.get_log_head()} called train_hypernetwork")
        try:
            shared.state.begin()
            shared.loaded_hypernetworks = []
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except AssertionError:
            shared.state.end()
            return models.TrainResponse(info=f"train embedding error: {error}")

    def get_memory(self):
        print(f"{api_log.get_log_head()} called get_memory")
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
            ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
        except Exception as err:
            ram = { 'error': f'{err}' }
        try:
            import torch
            if torch.cuda.is_available():
                s = torch.cuda.mem_get_info()
                system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
                s = dict(torch.cuda.memory_stats(shared.device))
                allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
                reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
                active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
                inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
                warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
                cuda = {
                    'system': system,
                    'active': active,
                    'allocated': allocated,
                    'reserved': reserved,
                    'inactive': inactive,
                    'events': warnings,
                }
            else:
                cuda = {'error': 'unavailable'}
        except Exception as err:
            cuda = {'error': f'{err}'}
        return models.MemoryResponse(ram=ram, cuda=cuda)

    # def launch(self, server_name, port):
    #     print(f"{api_log.get_log_head()} called launch")
    #     self.app.include_router(self.router)
    #     uvicorn.run(self.app, host=server_name, port=port)



class StableDiffusionWebuiWrapper:
    def __init__(self):
        self.api = ApiWrapper(queue_lock=queue_lock)

    def initialize(self):
        webui.initialize()

    def txt2img(self, model_name: str, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI) -> models.TextToImageResponse:
        self.api.refresh_checkpoints()
        selected_model = {"sd_model_checkpoint" : model_name}
        self.api.set_config(selected_model)
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
    output = webui.txt2img("MeinaMix.safetensors", input)
    image = output.images[0]
    pic = decode_base64_to_image(image)
    pic.save("1.jpg")
