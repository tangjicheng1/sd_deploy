def realesrgan_models_names():
    from . import realesrgan_model
    return [x.name for x in realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    from . import scripts

    return scripts.scripts_postproc.scripts


def sd_vae_items():
    from . import sd_vae

    return ["Automatic", "None"] + list(sd_vae.vae_dict)


def refresh_vae_list():
    from . import sd_vae

    sd_vae.refresh_vae_list()


def cross_attention_optimizations():
    from . import sd_hijack

    return ["Automatic"] + [x.title() for x in sd_hijack.optimizers] + ["None"]


