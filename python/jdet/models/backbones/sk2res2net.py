
import math
import copy
import jittor as jt
from jittor import init
from jittor import nn
from jdet.utils.registry import BACKBONES

def build_plugin_layer(*args, **kargs):
    pass

def build_norm_layer(cfg, num_features, postfix=''):
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    assert (layer_type == 'BN')
    norm_layer = nn.BatchNorm
    abbr = 'bn'
    assert isinstance(postfix, (int, str))
    name = (abbr + str(postfix))
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    layer = norm_layer(num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return (name, layer)


def build_conv_layer(cfg, *args, **kwargs):
    assert cfg is None
    cfg_ = dict(type='Conv2d')
    cfg_.pop('type')
    return nn.Conv2d(*args, **kwargs, **cfg_)


class ResLayer(nn.Sequential):

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), downsample_first=True, **kwargs):
        self.block = block
        downsample = None
        if ((stride != 1) or (inplanes != (planes * block.expansion))):
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, inplanes, (planes * block.expansion), kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, (planes * block.expansion))[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        if downsample_first:
            layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            inplanes = (planes * block.expansion)
            for _ in range(1, num_blocks):
                layers.append(block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        else:
            for _ in range((num_blocks - 1)):
                layers.append(block(inplanes=inplanes, planes=inplanes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
            layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super(ResLayer, self).__init__(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        super(BasicBlock, self).__init__()
        assert (dcn is None), 'Not implemented yet.'
        assert (plugins is None), 'Not implemented yet.'
        (self.norm1_name, norm1) = build_norm_layer(norm_cfg, planes, postfix=1)
        (self.norm2_name, norm2) = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        # self.add_module(self.norm1_name, norm1)
        # eval('self.'+self.norm1_name+'=norm1')
        self.bn1 = norm1
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        # self.add_module(self.norm2_name, norm2)
        exec('self.'+self.norm2_name+'=norm2')
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def execute(self, x):

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = nn.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            if (self.downsample is not None):
                identity = self.downsample(x)
            out += identity
            return out
        out = _inner_forward(x)
        out = nn.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, stage_number=0, expansion=4, plugins=None):
        super(Bottleneck, self).__init__()
        assert (style in ['pytorch', 'caffe'])
        assert ((dcn is None) or isinstance(dcn, dict))
        assert ((plugins is None) or isinstance(plugins, list))
        if (plugins is not None):
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(((p['position'] in allowed_position) for p in plugins))
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = (dcn is not None)
        self.plugins = plugins
        self.with_plugins = (plugins is not None)
        self.expansion = expansion
        if self.with_plugins:
            self.after_conv1_plugins = [plugin['cfg'] for plugin in plugins if (plugin['position'] == 'after_conv1')]
            self.after_conv2_plugins = [plugin['cfg'] for plugin in plugins if (plugin['position'] == 'after_conv2')]
            self.after_conv3_plugins = [plugin['cfg'] for plugin in plugins if (plugin['position'] == 'after_conv3')]
        if (self.style == 'pytorch'):
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        (self.norm1_name, norm1) = build_norm_layer(norm_cfg, planes, postfix=1)
        (self.norm2_name, norm2) = build_norm_layer(norm_cfg, planes, postfix=2)
        (self.norm3_name, norm3) = build_norm_layer(norm_cfg, (planes * self.expansion), postfix=3)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        # self.add_module(self.norm1_name, norm1)
        self.bn1 = norm1
        # eval('self.'+self.norm1_name+' = norm1')
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if ((not self.with_dcn) or fallback_on_stride):
            self.conv2 = build_conv_layer(conv_cfg, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        else:
            assert (self.conv_cfg is None), 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(dcn, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        # self.add_module(self.norm2_name, norm2)
        # eval('self.'+self.norm2_name+'=norm2')
        self.bn2 = norm2
        self.conv3 = build_conv_layer(conv_cfg, planes, (planes * self.expansion), kernel_size=1, bias=False)
        # self.add_module(self.norm3_name, norm3)
        # eval('self.'+self.norm3_name+'=norm3')
        self.bn3 = norm3
        self.relu = nn.ReLU()
        self.downsample = downsample
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins((planes * self.expansion), self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            (name, layer) = build_plugin_layer(plugin, in_channels=in_channels, postfix=plugin.pop('postfix', ''))
            assert (not hasattr(self, name)), f'duplicate plugin {name}'
            # self.add_module(name, layer)
            exec('self.'+name+'=layer')
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def execute(self, x):
        print('btnk execute')

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = nn.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            out = self.conv2(out)
            out = self.norm2(out)
            out = nn.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)
            if (self.downsample is not None):
                identity = self.downsample(x)
            out += identity
            return out
        out = _inner_forward(x)
        out = nn.relu(out)
        return out

class ResNet(nn.Module):
    arch_settings = {18: (BasicBlock, (2, 2, 2, 2)), 34: (BasicBlock, (3, 4, 6, 3)), 50: (Bottleneck, (3, 4, 6, 3)), 101: (Bottleneck, (3, 4, 23, 3)), 152: (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, in_channels=3, stem_channels=64, base_channels=64, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), style='pytorch', deep_stem=False, avg_down=False, frozen_stages=(- 1), conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=False, dcn=None, stage_with_dcn=(False, False, False, False), plugins=None, zero_init_residual=True):
        super(ResNet, self).__init__()
        if (depth not in self.arch_settings):
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        if (stem_channels is None):
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert ((num_stages >= 1) and (num_stages <= 4))
        self.strides = strides
        self.dilations = dilations
        assert (len(strides) == len(dilations) == num_stages)
        self.out_indices = out_indices
        assert (max(out_indices) < num_stages)
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if (dcn is not None):
            assert (len(stage_with_dcn) == num_stages)
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        (self.block, stage_blocks) = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self._make_stem_layer(in_channels, stem_channels)
        self.res_layers = []
        for (i, num_blocks) in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = (self.dcn if self.stage_with_dcn[i] else None)
            if (plugins is not None):
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = (base_channels * (2 ** i))
            res_layer = self.make_res_layer(block=self.block, inplanes=self.inplanes, planes=planes, num_blocks=num_blocks, stride=stride, dilation=dilation, style=self.style, avg_down=self.avg_down, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, plugins=stage_plugins, stage_number=i)
            self.inplanes = (planes * self.block.expansion)
            layer_name = f'layer{(i + 1)}'
            # self.add_module(layer_name, res_layer)
            exec('self.'+layer_name+'=res_layer')
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = ((self.block.expansion * base_channels) * (2 ** (len(self.stage_blocks) - 1)))

    def make_stage_plugins(self, plugins, stage_idx):
        print('in make_stage_plugins')
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert ((stages is None) or (len(stages) == self.num_stages))
            if ((stages is None) or stages[stage_idx]):
                stage_plugins.append(plugin)
        return stage_plugins

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(build_conv_layer(self.conv_cfg, in_channels, (stem_channels // 2), kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, (stem_channels // 2))[1], nn.ReLU(), build_conv_layer(self.conv_cfg, (stem_channels // 2), (stem_channels // 2), kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, (stem_channels // 2))[1], nn.ReLU(), build_conv_layer(self.conv_cfg, (stem_channels // 2), stem_channels, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, stem_channels)[1], nn.ReLU())
        else:
            self.conv1 = build_conv_layer(self.conv_cfg, in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            (self.norm1_name, norm1) = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            # eval('self.'+self.norm1_name+' = norm1')
            self.bn1 = norm1
            # self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU()
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='maximum')

    def _freeze_stages(self):
        if (self.frozen_stages >= 0):
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        for i in range(1, (self.frozen_stages + 1)):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def execute(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = nn.relu(x)
        x = self.maxpool(x)
        outs = []
        for (i, layer_name) in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if (i in self.out_indices):
                outs.append(x)
        if (len(outs) == 1):
            return outs[0]
        return tuple(outs)

    def train(self):
        super(ResNet, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm):
                    m.eval()

class SKLayer(nn.Module):

    def __init__(self, inplanes, ratio=8) -> None:
        super(SKLayer, self).__init__()
        planes = max(32, (inplanes // ratio))
        self.sk_proj = nn.Sequential(nn.Conv(inplanes, planes, 1, bias=False), nn.BatchNorm(planes), nn.ReLU(), nn.Conv(planes, inplanes, 1, bias=True))
        self.sk_downsample = None
        self.alpha = jt.array(jt.zeros((1, 1, 1, 1)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def execute(self, x, fx):
        a = self.alpha.sigmoid()
        fuse = ((a * self.avg_pool(x)) + ((1.0 - a) * self.avg_pool(fx)))
        sk = self.sk_proj(fuse).sigmoid()
        out = (((fx - x) * sk) + x)
        return out

class Bottle2neck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, scales=4, base_width=32, base_channels=64, cardinality=8, ratio=8, avg_down_after=False, stage_number=0, stage_type='normal', **kwargs):
        super(Bottle2neck, self).__init__(inplanes, planes, **kwargs)
        assert (scales > 1), 'SK2Net degenerates to ResNet when scales = 1.'
        width = int(math.floor((self.planes * (base_width / base_channels))))
        (self.norm1_name, norm1) = build_norm_layer(self.norm_cfg, (width * scales), postfix=1)
        (self.norm3_name, norm3) = build_norm_layer(self.norm_cfg, (self.planes * self.expansion), postfix=3)
        self.conv1 = build_conv_layer(self.conv_cfg, self.inplanes, (width * scales), kernel_size=1, stride=self.conv1_stride, bias=False)
        # eval('self.'+self.norm1_name+' = norm1')
        self.bn1 = norm1
        # self.add_module(self.norm1_name, norm1)
        self.avg_down_after = avg_down_after
        self.pool = None
        if ((stage_type == 'stage') and (self.conv2_stride != 1)):
            self.pool = nn.Pool(3, stride=self.conv2_stride, padding=1, op='mean')
        convs = []
        bns = []
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if ((not self.with_dcn) or fallback_on_stride):
            for i in range((scales - 1)):
                convs.append(build_conv_layer(self.conv_cfg, width, width, kernel_size=3, groups=cardinality, stride=1, padding=self.dilation, dilation=self.dilation, bias=False))
                bns.append(build_norm_layer(self.norm_cfg, width, postfix=(i + 1))[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        else:
            assert (self.conv_cfg is None), 'conv_cfg must be None for DCN'
            for i in range((scales - 1)):
                convs.append(build_conv_layer(self.dcn, width, width, kernel_size=3, groups=cardinality, stride=1, padding=self.dilation, dilation=self.dilation, bias=False))
                bns.append(build_norm_layer(self.norm_cfg, width, postfix=(i + 1))[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        self.sk = SKLayer((width * scales), ratio)
        self.conv3 = build_conv_layer(self.conv_cfg, (width * scales), (self.planes * self.expansion), kernel_size=1, bias=False)
        self.bn3 = norm3
        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def execute(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = nn.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            if (self.pool and (not self.avg_down_after)):
                out = self.pool(out)
            spx = jt.split(out, self.width, 1)
            sp = self.convs[0](copy.deepcopy(spx[0]))
            sp = nn.relu(self.bns[0](sp))
            out = sp
            for i in range(1, (self.scales - 1)):
                if (self.stage_type == 'stage'):
                    sp = spx[i]
                else:
                    sp = (sp + spx[i])
                sp = self.convs[i](copy.deepcopy(sp))
                sp = nn.relu(self.bns[i](sp))
                out = jt.concat((out, sp), 1)
            out = jt.concat((out, spx[(self.scales - 1)]), 1)
            old_out = jt.concat(spx[:self.scales], 1)
            out = self.sk(old_out, out)
            if (self.pool and self.avg_down_after):
                out = self.pool(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)
            if (self.downsample is not None):
                identity = self.downsample(x)
            out += identity
            return out
        out = _inner_forward(x)
        out = nn.relu(out)
        return out

class SK2Layer(nn.Sequential):

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, avg_down=True, conv_cfg=None, norm_cfg=dict(type='BN'), scales=4, base_width=32, **kwargs):
        self.block = block
        downsample = None
        if ((stride != 1) or (inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False), build_conv_layer(conv_cfg, inplanes, (planes * block.expansion), kernel_size=1, stride=1, bias=False), build_norm_layer(norm_cfg, (planes * block.expansion))[1])
        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, scales=scales, base_width=base_width, stage_type='stage', **kwargs))
        inplanes = (planes * block.expansion)
        for i in range(1, num_blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, scales=scales, base_width=base_width, **kwargs))
        super(SK2Layer, self).__init__(*layers)

class SK2Res2Net(ResNet):
    arch_settings = {50: (Bottle2neck, (3, 4, 6, 3)), 101: (Bottle2neck, (3, 4, 23, 3)), 152: (Bottle2neck, (3, 8, 36, 3))}

    def __init__(self, scales=4, base_width=26, cardinality=1, ratio=8, style='pytorch', avg_down_after=False, deep_stem=True, avg_down=True, num_classes=1000, args=None, **kwargs):
        self.scales = scales
        self.base_width = base_width
        self.ratio = ratio
        self.cardinality = cardinality
        self.avg_down_after = avg_down_after
        super(SK2Res2Net, self).__init__(style=style, deep_stem=deep_stem, avg_down=avg_down, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear((512 * Bottle2neck.expansion), num_classes)

    def make_res_layer(self, **kwargs):
        return SK2Layer(scales=self.scales, base_width=self.base_width, base_channels=self.base_channels, cardinality=self.cardinality, ratio=self.ratio, avg_down_after=self.avg_down_after, **kwargs)

    def execute(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = nn.relu(x)
        x = self.maxpool(x)
        res = []
        for (i, layer_name) in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            res.append(x)
        return res

@BACKBONES.register_module()
def sk2res2net101(pretrained=False):
    model = SK2Res2Net(depth=101, num_stages=4, scales=4, base_width=26, ratio=4, avg_down_after=True, deep_stem=True, avg_down=True, num_classes=10)
    if pretrained:
        model_path = '/opt/data/private/LYX/data/pretrained/sk2res2net101_epoch_300.pth'
        model.load_parameters(jt.load(model_path))
    return model

