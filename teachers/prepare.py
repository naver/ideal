

from types import MethodType

import torch.nn as nn
from timm.models.vision_transformer import Attention


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def attention_forward(self, x, **kwargs):
    """Attention forward with hookable matmul modules.

    Extra keyword arguments passed by newer timm versions (e.g. ``attn_mask``,
    ``is_causal``) are ignored: teachers are run unmasked during generation.
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def inject_attn_matmuls(model):
    """Inject MatMul modules & swap forward for every timm Attention block.

    Only needed for timm-based teachers (dBOT-ft inherits its attention
    blocks from timm's VisionTransformer); the DINO/iBOT/DeiT-3
    implementations define matmul1/matmul2 natively in their own
    Attention classes.
    """
    for _name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)


def prepare_teacher_models(teachers, device):
    """Move teachers to *device* and inject hookable matmuls."""
    prepared = {}
    for teacher_name, teacher in teachers.items():
        t = teacher.to(device).eval()
        inject_attn_matmuls(t)
        prepared[teacher_name] = t
    return prepared
