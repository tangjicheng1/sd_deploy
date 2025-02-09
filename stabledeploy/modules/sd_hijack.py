import torch
from torch.nn.functional import silu
from types import MethodType

from .textual_inversion import textual_inversion
from . import devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr
from .hypernetworks import hypernetwork
from .shared import cmd_opts

import stabledeploy.ldm.modules.attention
import stabledeploy.ldm.modules.diffusionmodules.model
import stabledeploy.ldm.modules.diffusionmodules.openaimodel
import stabledeploy.ldm.models.diffusion.ddim
import stabledeploy.ldm.models.diffusion.plms
import stabledeploy.ldm.modules.encoders.modules

attention_CrossAttention_forward = stabledeploy.ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = stabledeploy.ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = stabledeploy.ldm.modules.diffusionmodules.model.AttnBlock.forward

# new memory efficient cross attention blocks do not support hypernets and we already
# have memory efficient cross attention anyway, so this disables SD2.0's memory efficient cross attention
stabledeploy.ldm.modules.attention.MemoryEfficientCrossAttention = stabledeploy.ldm.modules.attention.CrossAttention
stabledeploy.ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = stabledeploy.ldm.modules.attention.CrossAttention

# silence new console spam from SD2
stabledeploy.ldm.modules.attention.print = lambda *args: None
stabledeploy.ldm.modules.diffusionmodules.model.print = lambda *args: None

optimizers = []
current_optimizer: sd_hijack_optimizations.SdOptimization = None


def list_optimizers():
    new_optimizers = script_callbacks.list_optimizers_callback()

    new_optimizers = [x for x in new_optimizers if x.is_available()]

    new_optimizers = sorted(new_optimizers, key=lambda x: x.priority, reverse=True)

    optimizers.clear()
    optimizers.extend(new_optimizers)


def apply_optimizations():
    global current_optimizer

    undo_optimizations()

    if len(optimizers) == 0:
        # a script can access the model very early, and optimizations would not be filled by then
        current_optimizer = None
        return ''

    stabledeploy.ldm.modules.diffusionmodules.model.nonlinearity = silu
    stabledeploy.ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None

    selection = shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.title() == selection]), None)

    if selection == "None":
        matching_optimizer = None
    elif selection == "Automatic" and shared.cmd_opts.disable_opt_split_attention:
        matching_optimizer = None
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]

    if matching_optimizer is not None:
        print(f"Applying optimization: {matching_optimizer.name}... ", end='')
        matching_optimizer.apply()
        print("done.")
        current_optimizer = matching_optimizer
        return current_optimizer.name
    else:
        return ''


def undo_optimizations():
    stabledeploy.ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    stabledeploy.ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    stabledeploy.ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""

    pass


def weighted_loss(sd_model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
    loss = sd_model._old_get_loss(pred, target, mean=False)

    #Check if we have weights available
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
        sd_model._custom_loss_weight = w

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        #Run the standard forward function, but with the patched 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss
            del sd_model._old_get_loss

def apply_weighted_forward(sd_model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass


class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    embedding_db = textual_inversion.EmbeddingDatabase()

    def __init__(self):
        self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)

    def apply_optimizations(self):
        try:
            self.optimization_method = apply_optimizations()
        except Exception as e:
            errors.display(e, "applying cross attention optimization")
            undo_optimizations()

    def hijack(self, m):
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            model_embeddings = m.cond_stage_model.roberta.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
            m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == stabledeploy.ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == stabledeploy.ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
            m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        apply_weighted_forward(m)
        if m.cond_stage_key == "edit":
            sd_hijack_unet.hijack_ddpm_edit()

        self.apply_optimizations()

        self.clip = m.cond_stage_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

    def undo_hijack(self, m):
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            m.cond_stage_model = m.cond_stage_model.wrapped

        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped

        undo_optimizations()
        undo_weighted_forward(m)

        self.apply_circular(False)
        self.layers = None
        self.clip = None

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return

        self.circular_enabled = enable

        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []

    def get_prompt_lengths(self, text):
        if self.clip is None:
            return "-", "-"

        _, token_count = self.clip.process_texts([text])

        return token_count, self.clip.get_target_prompt_token_count(token_count)

    def redo_hijack(self, m):
        self.undo_hijack(m)
        self.hijack(m)


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = devices.cond_cast_unet(embedding.vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)


def add_circular_option_to_conv_2d():
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


model_hijack = StableDiffusionModelHijack()


def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    if type(attr) == torch.Tensor:
        if attr.device != devices.device:
            attr = attr.to(device=devices.device, dtype=(torch.float32 if devices.device.type == 'mps' else None))

    setattr(self, name, attr)


stabledeploy.ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
stabledeploy.ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer
