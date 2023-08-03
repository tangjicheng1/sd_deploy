from torch.utils.checkpoint import checkpoint

import stabledeploy.ldm.modules.attention
import stabledeploy.ldm.modules.diffusionmodules.openaimodel


def BasicTransformerBlock_forward(self, x, context=None):
    return checkpoint(self._forward, x, context)


def AttentionBlock_forward(self, x):
    return checkpoint(self._forward, x)


def ResBlock_forward(self, x, emb):
    return checkpoint(self._forward, x, emb)


stored = []


def add():
    if len(stored) != 0:
        return

    stored.extend([
        stabledeploy.ldm.modules.attention.BasicTransformerBlock.forward,
        stabledeploy.ldm.modules.diffusionmodules.openaimodel.ResBlock.forward,
        stabledeploy.ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward
    ])

    stabledeploy.ldm.modules.attention.BasicTransformerBlock.forward = BasicTransformerBlock_forward
    stabledeploy.ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = ResBlock_forward
    stabledeploy.ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = AttentionBlock_forward


def remove():
    if len(stored) == 0:
        return

    stabledeploy.ldm.modules.attention.BasicTransformerBlock.forward = stored[0]
    stabledeploy.ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = stored[1]
    stabledeploy.ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = stored[2]

    stored.clear()

