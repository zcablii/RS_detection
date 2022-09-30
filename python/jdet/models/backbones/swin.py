import jittor as jt
from jittor import nn

from jdet.utils.registry import BACKBONES
from .jimm import swin_large_patch4_window12_384_in22k,swin_base_patch4_window12_384_in22k

@BACKBONES.register_module()
def swin_large(pretrained=True, **kwargs):
    model = swin_large_patch4_window12_384_in22k(pretrained = pretrained, **kwargs)
    return model

@BACKBONES.register_module()
def swin_base(pretrained=True, **kwargs):
    model = swin_base_patch4_window12_384_in22k(pretrained = pretrained, **kwargs)
    return model
