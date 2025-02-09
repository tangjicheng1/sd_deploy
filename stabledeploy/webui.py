from __future__ import annotations

import os
import sys
import time
import importlib
import signal
import re
import warnings
import json
from threading import Thread
from typing import Iterable

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging

logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from stabledeploy.modules import sd_models, paths, timer, import_hook, errors  # noqa: F401

startup_timer = timer.Timer()

import torch
import pytorch_lightning   # noqa: F401 # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")


startup_timer.record("import torch")

import gradio
startup_timer.record("import gradio")

import stabledeploy.ldm.modules.encoders.modules  # noqa: F401
startup_timer.record("import ldm")

from stabledeploy.modules import extra_networks
from stabledeploy.modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, queue_lock  # noqa: F401

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from stabledeploy.modules import shared, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks, config_states
from stabledeploy.modules import codeformer_model as codeformer
from stabledeploy.modules import face_restoration
from stabledeploy.modules import gfpgan_model as gfpgan
from stabledeploy.modules import img2img

from stabledeploy.modules import lowvram
from stabledeploy.modules import scripts
from stabledeploy.modules import sd_hijack
from stabledeploy.modules import sd_hijack_optimizations
from stabledeploy.modules import sd_vae
from stabledeploy.modules import txt2img
from stabledeploy.modules import script_callbacks
from stabledeploy.modules.textual_inversion import textual_inversion
from stabledeploy.modules import progress

from stabledeploy.modules import ui
from stabledeploy.modules import modelloader
from stabledeploy.modules import shared
from stabledeploy.modules.hypernetworks import hypernetwork

startup_timer.record("other imports")


if shared.cmd_opts.server_name:
    server_name = shared.cmd_opts.server_name
else:
    server_name = "0.0.0.0" if shared.cmd_opts.listen else None

def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "2.0.1"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.20"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())


def restore_config_state_file():
    config_state_file = shared.opts.restore_config_state_file
    if config_state_file == "":
        return

    shared.opts.restore_config_state_file = ""
    shared.opts.save(shared.config_filename)

    if os.path.isfile(config_state_file):
        print(f"*** About to restore extension state from file: {config_state_file}")
        with open(config_state_file, "r", encoding="utf-8") as f:
            config_state = json.load(f)
            config_states.restore_extension_config(config_state)
        startup_timer.record("restore extension config")
    elif config_state_file:
        print(f"!!! Config state backup not found: {config_state_file}")


def get_gradio_auth_creds() -> Iterable[tuple[str, ...]]:
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    def process_credential_line(s) -> tuple[str, ...] | None:
        s = s.strip()
        if not s:
            return None
        return tuple(s.split(':', 1))

    if shared.cmd_opts.gradio_auth:
        for cred in shared.cmd_opts.gradio_auth.split(','):
            cred = process_credential_line(cred)
            if cred:
                yield cred

    if shared.cmd_opts.gradio_auth_path:
        with open(shared.cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                for cred in line.strip().split(','):
                    cred = process_credential_line(cred)
                    if cred:
                        yield cred


def configure_sigint_handler():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
        signal.signal(signal.SIGINT, sigint_handler)


def configure_opts_onchange():
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    shared.opts.onchange("cross_attention_optimization", wrap_queued_call(lambda: sd_hijack.model_hijack.redo_hijack(shared.sd_model)), call=False)
    startup_timer.record("opts onchange")


def initialize():
    configure_sigint_handler()
    check_versions()
    modelloader.cleanup_models()
    configure_opts_onchange()

    sd_models.setup_model()
    startup_timer.record("setup SD model")

    codeformer.setup_model(shared.cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    gfpgan.setup_model(shared.cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    sd_samplers.set_samplers()
    extensions.list_extensions()
    startup_timer.record("list extensions")

    restore_config_state_file()

    if shared.cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    sd_models.list_models()
    startup_timer.record("list SD models")

    localization.list_localizations(shared.cmd_opts.localizations_dir)

    scripts.load_scripts()
    startup_timer.record("load scripts")

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")
    textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
    sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """

        shared.sd_model  # noqa: B018

        if sd_hijack.current_optimizer is None:
            sd_hijack.apply_optimizations()

    Thread(target=load_model).start()

    shared.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")


def setup_middleware(app):
    app.middleware_stack = None  # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if shared.cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = shared.cmd_opts.cors_allow_origins.split(',')
    if shared.cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = shared.cmd_opts.cors_allow_origins_regex
    app.add_middleware(CORSMiddleware, **cors_options)


def create_api(app):
    from stabledeploy.modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def api_only():
    initialize()

    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)

    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if shared.cmd_opts.listen else "127.0.0.1", port=shared.cmd_opts.port if shared.cmd_opts.port else 7861)


def stop_route(request):
    shared.state.server_command = "stop"
    return Response("Stopping.")


def webui():
    launch_api = shared.cmd_opts.api
    initialize()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        if not shared.cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(get_gradio_auth_creds()) or None

        # this restores the missing /docs endpoint
        if launch_api and not hasattr(FastAPI, 'original_setup'):
            # TODO: replace this with `launch(app_kwargs=...)` if https://github.com/gradio-app/gradio/pull/4282 gets merged
            def fastapi_setup(self):
                self.docs_url = "/docs"
                self.redoc_url = "/redoc"
                self.original_setup()

            FastAPI.original_setup = FastAPI.setup
            FastAPI.setup = fastapi_setup

        app, local_url, share_url = shared.demo.launch(
            share=shared.cmd_opts.share,
            server_name=server_name,
            server_port=shared.cmd_opts.port,
            ssl_keyfile=shared.cmd_opts.tls_keyfile,
            ssl_certfile=shared.cmd_opts.tls_certfile,
            ssl_verify=shared.cmd_opts.disable_tls_verify,
            debug=shared.cmd_opts.gradio_debug,
            auth=gradio_auth_creds,
            inbrowser=shared.cmd_opts.autolaunch,
            prevent_thread_lock=True,
            allowed_paths=shared.cmd_opts.gradio_allowed_path,
        )
        if shared.cmd_opts.add_stop_route:
            app.add_route("/_stop", stop_route, methods=["POST"])

        # after initial launch, disable --autolaunch for subsequent restarts
        shared.cmd_opts.autolaunch = False

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        setup_middleware(app)

        progress.setup_progress_api(app)
        ui.setup_ui_api(app)

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        script_callbacks.app_started_callback(shared.demo, app)
        startup_timer.record("scripts app_started_callback")

        print(f"Startup time: {startup_timer.summary()}.")

        if shared.cmd_opts.subpath:
            redirector = FastAPI()
            redirector.get("/")
            gradio.mount_gradio_app(redirector, shared.demo, path=f"/{shared.cmd_opts.subpath}")

        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break
        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize_rest(reload_script_modules=True)

        script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
        sd_hijack.list_optimizers()
        startup_timer.record("scripts list_optimizers")


if __name__ == "__main__":
    if shared.cmd_opts.nowebui:
        api_only()
    else:
        webui()
