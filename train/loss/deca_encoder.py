import sys, os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import resnet

class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        return features

def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                # print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            # print('copy param {} failed'.format(k))
            continue
if __name__ == '__main__':
    E_flame = ResnetEncoder(outsize=236).to('cuda') 
    model_path = 'DECA/data/deca_model.tar'
    print(f'trained model found. load {model_path}')
    checkpoint = torch.load(model_path)
    copy_state_dict(E_flame.state_dict(), checkpoint['E_flame'])
    E_flame.eval()
    x = torch.randn(4,3,224,224).cuda()
    x2 = torch.randn(4,3,224,224).cuda()
    criterionFeat = torch.nn.L1Loss()
    y = E_flame(x)
    y2 = E_flame(x2)
    l = criterionFeat(x, x2)
    print(l)