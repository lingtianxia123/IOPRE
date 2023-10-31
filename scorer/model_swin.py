import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.hub import load_state_dict_from_url
import warnings
from .swin_transformer import SwinTransformer

urls = {
    'swin_t': "https://download.pytorch.org/models/swin_t-704ceda3.pth",
    'swin_s': "https://download.pytorch.org/models/swin_s-5e29d889.pth",
    'swin_b': "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
}

def swin_base(arch='swin_t', input_dim=4, num_classes=2):
    url = urls[arch]
    if arch == 'swin_t':
        model = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
            num_classes=1000,
        )
    elif arch == 'swin_s':
        model = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.3,
            num_classes=1000,
        )
    elif arch == 'swin_b':
        model = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=[7, 7],
            stochastic_depth_prob=0.5,
            num_classes=1000,
        )
    else:
        raise ValueError(f'SwinTransformer {arch} not supported')
    middle_dim = model.features[0][0].out_channels
    model.features[0][0] = nn.Conv2d(input_dim, middle_dim, kernel_size=(4, 4), stride=(4, 4))

    # weight
    pretrained_state_dict = load_state_dict_from_url(url, progress=True)
    conv_proj = pretrained_state_dict['features.0.0.weight']
    new = torch.zeros(middle_dim, 1, 4, 4)
    for i, output_channel in enumerate(conv_proj):
        new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
    new = new.repeat([1, input_dim - 3, 1, 1])
    pretrained_state_dict['features.0.0.weight'] = torch.cat((conv_proj, new), dim=1)

    model.load_state_dict(pretrained_state_dict)
    print('loaded pretrained {} from {}'.format(arch, url))

    model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    return model

class ObjectPlaceNet(nn.Module):
    def __init__(self, arch='swin_t', num_classes=2):
        super(ObjectPlaceNet, self).__init__()

        backbone = swin_base(arch=arch, input_dim=4,  num_classes=num_classes)
        self.fc_label = nn.Linear(backbone.head.in_features, num_classes, bias=True)
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.feat_dim = self.fc_label.in_features

    def forward(self, img, msk):
        img_cat = torch.cat([img, msk], dim=1)

        feat = self.backbone(img_cat)
        label = self.fc_label(feat)

        out = {}
        out['label'] = label
        out['feat'] = feat
        return out


def build_Scorer(args):
    model = ObjectPlaceNet(arch=args.scorer, num_classes=2)

    if args.scorer_weight:
        model.load_state_dict(torch.load(args.scorer_weight)['model'])
        print('Scorer load pretrained weights from ', args.scorer_weight)
    else:
        warnings.warn('Scorer weight is None!')

    model.requires_grad_(False)
    model = model.eval()

    return model