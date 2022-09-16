
import jittor as jt
from jittor import init
from jittor import nn
import math
import warnings

from jdet.utils.registry import BACKBONES

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    def norm_cdf(x):
        return ((1.0 + math.erf((x / math.sqrt(2.0)))) / 2.0)
    if ((mean < (a - (2 * std))) or (mean > (b + (2 * std)))):
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with jt.no_grad():
        l = norm_cdf(((a - mean) / std))
        u = norm_cdf(((b - mean) / std))
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor = tensor.erfinv()
        tensor = tensor.multiply((std * math.sqrt(2.0)))
        tensor = tensor.add(mean)
        tensor = tensor.clamp(min_v=a, max_v=b)
        return tensor

def drop_path(x, drop_prob: float=0.0, training: bool=False):
    if ((drop_prob == 0.0) or (not training)):
        return x
    keep_prob = (1 - drop_prob)
    shape = ((x.shape[0],) + ((1,) * (x.ndim - 1)))
    random_tensor = (keep_prob + jt.rand(shape, dtype=x.dtype, device=x.device))
    random_tensor.floor_()
    output = (x.div(keep_prob) * random_tensor)
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-06):
        super().__init__()
        self.dwconv = nn.Conv(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, (4 * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear((4 * dim), dim)
        self.gamma = layer_scale_init_value * jt.ones((dim)) if layer_scale_init_value > 0 else None
        self.drop_path = (DropPath(drop_path) if (drop_path > 0.0) else nn.Identity())

    def execute(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute((0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if (self.gamma is not None):
            x = (self.gamma * x)
        x = x.permute((0, 3, 1, 2))
        x = (input + self.drop_path(x))
        return x

class ConvNeXt(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], return_stages=["layer1","layer2","layer3","layer4"], drop_path_rate=0.0, layer_scale_init_value=1e-06, norm_eval=True,frozen_stages=-1,head_init_scale=1.0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv(in_chans, dims[0], 4, stride=4), LayerNorm(dims[0], eps=1e-06, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-06, data_format='channels_first'), nn.Conv(dims[i], dims[(i + 1)], 2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[(cur + j)], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.return_stages = [eval(i[-1]) for i in return_stages]
        self.norm = nn.LayerNorm(dims[(- 1)], eps=1e-06)
        self.head = nn.Linear(dims[(- 1)], num_classes)
        self.apply(self._init_weights)
        self.norm_eval = norm_eval 
        self.frozen_stages = frozen_stages
        # self.head.weight.data = self.head.weight.data.multiply(head_init_scale)
        # self.head.bias.data = self.head.bias.data.multiply(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            init.constant_(m.bias, value=0)

    def forward_features(self, x):
        output = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i+1 in self.return_stages:
                output.append(x)
        return tuple(output) #  self.norm(x.mean([(- 2), (- 1)]))

    def execute(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x
    
    def _freeze_stages(self):

        for i in range(1, self.frozen_stages + 1):
            down_m = self.downsample_layers[i]   
            down_m.eval()
            for param in down_m.parameters():
                param.stop_grad() 

            stage_m = self.stages[i]
            stage_m.eval()
            for param in stage_m.parameters():
                param.stop_grad()  

    def train(self):
        super(ConvNeXt, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.LayerNorm):
                    m.eval()



class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = jt.array(jt.ones(normalized_shape))
        self.bias = jt.array(jt.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if (self.data_format not in ['channels_last', 'channels_first']):
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def execute(self, x):
        if (self.data_format == 'channels_last'):
            return nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif (self.data_format == 'channels_first'):
            u = x.mean(1, keepdims=True)
            s = (x - u).pow(2).mean(1, keepdims=True)
            x = ((x - u) / jt.sqrt((s + self.eps)))
            x = ((self.weight[:, None, None] * x) + self.bias[:, None, None])
            return x
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@BACKBONES.register_module()
def Convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    # ckpt_path = '/media/data3/lyx/Detection/pretrained/convnext_ema.pth'
    # model.load_state_dict(jt.load(ckpt_path)['model'])
    if pretrained:
        url = (model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k'])
        checkpoint = jt.load(url)
        model.load_state_dict(checkpoint['model'])
    return model


@BACKBONES.register_module()
def Convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = (model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k'])
        checkpoint = jt.load(url)
        model.load_state_dict(checkpoint['model'])
    return model

@BACKBONES.register_module()
def Convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = (model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k'])
        checkpoint = jt.load(url)
        model.load_state_dict(checkpoint['model'])
    return model

@BACKBONES.register_module()
def Convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        print('loading model convNext xlarge')
        ckpt_path = '/opt/data/private/LYX/data/pretrained/convnext_large_1k_384.pth'
        state_dict = jt.load(ckpt_path)['model']
 
        model.load_state_dict(state_dict)
       
    return model

@BACKBONES.register_module()
def Convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        print('loading model convNext xlarge')
        ckpt_path = '/opt/data/private/LYX/data/pretrained/convnext_xlarge_22k_1k_384_ema.pth'
        state_dict = jt.load(ckpt_path)['model']
        # print(state_dict)
        # for k in state_dict.keys():
        #     state_dict[k] = state_dict[k]/20
        model.load_state_dict(state_dict)
        # assert in_22k, 'only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True'
        # url = model_urls['convnext_xlarge_22k']
        # checkpoint = jt.load(url)
        # model.load_state_dict(checkpoint['model'])
    return model
