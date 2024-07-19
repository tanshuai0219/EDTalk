from transformers import BertTokenizer, BertModel

from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
 
class AudioExpClassifier(nn.Module):
    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self):
        super(AudioExpClassifier, self).__init__()
        self.audio2exp = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.audio2exp_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 10))

    def forward(self, audio_feature):

        audio2emo_feat = self.audio2exp(audio_feature)
        audio2emo_embed = self.audio2exp_embed(audio2emo_feat) # torch.Size([24, 30])

        return audio2emo_embed
 
class TextAudioExpClassifier(nn.Module):

    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self):
        super(TextAudioExpClassifier, self).__init__()

        # 定义 Bert 模型
        self.text_pooler = nn.Sequential(nn.Linear(768, 768), nn.Tanh())
        self.text2exp = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.text2exp_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 256))
        # 外接全连接层
        

        self.audio2exp = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 512))
        self.audio2exp_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 256))

        self.text_audio_mlp = nn.Sequential(nn.ReLU(), nn.Linear(512, 20))


    def forward(self, res, audio_feature, is_train=True):
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        text_feature = self.text_pooler(res[:,0])
        text2emo_feat = self.text2exp(text_feature) # torch.Size([1, 7])
        text2emo_embed = self.text2exp_embed(text2emo_feat) # torch.Size([24, 30])

        audio2emo_feat = self.audio2exp(audio_feature)
        audio2emo_embed = self.audio2exp_embed(audio2emo_feat) # torch.Size([24, 30])

        # if is_train:
        #     r = random.random()
        #     if r > 0.8:
        #         pass
        #     elif r >= 0.7:
        #         text2emo_embed = torch.zeros_like(text2emo_embed)
        #     elif r >= 0.6:
        #         audio2emo_embed = torch.zeros_like(audio2emo_embed)
        #     elif r >= 0.4:
        #         video2exp_embed = torch.zeros_like(video2exp_embed)
        #     elif r >= 0.2:
        #         video2exp_embed = torch.zeros_like(video2exp_embed)
        #         audio2emo_embed = torch.zeros_like(audio2emo_embed)
        #     elif r >= 0.1:
        #         video2exp_embed = torch.zeros_like(video2exp_embed)
        #         text2emo_embed = torch.zeros_like(text2emo_embed)
        #     elif r >= 0.0:
        #         audio2emo_embed = torch.zeros_like(audio2emo_embed)
        #         text2emo_embed = torch.zeros_like(text2emo_embed)
        # else:
        #     video2exp_embed = torch.zeros_like(video2exp_embed)

        text_audio_exp_embed = torch.cat([text2emo_embed, audio2emo_embed], dim=-1)

        text_audio_exp_embed = self.text_audio_video_mlp(text_audio_exp_embed)
        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析

        return text_audio_exp_embed
 
    def forward_v3(self, res, audio_feature, video2emo=None, label = None, is_train=True, using_text = True):
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        text_feature = self.text_pooler(res[:,0])
        text2emo_feat = self.text2exp(text_feature) # torch.Size([1, 7])
        text2emo_embed = self.text2exp_embed(text2emo_feat) # torch.Size([24, 30])
        if using_text== False:
            text2emo_embed = torch.zeros_like(text2emo_embed)
        if label!= None:
            label = label.unsqueeze(1).expand_as(text2emo_embed)
            text2emo_embed = text2emo_embed*label

        audio2emo_feat = self.audio2exp(audio_feature)
        audio2emo_embed = self.audio2exp_embed(audio2emo_feat) # torch.Size([24, 30])

        if video2emo!=None:
            video2emo_feat = self.video2exp(video2emo)
            video2exp_embed = self.video2exp_embed(video2emo_feat)
        else:
            video2exp_embed = torch.zeros_like(audio2emo_embed)

        if is_train:
            r = random.random()
            if r > 0.5:
                pass
            elif r >= 0.25:
                audio2emo_embed = torch.zeros_like(audio2emo_embed)
            else:
                video2exp_embed = torch.zeros_like(video2exp_embed)

        else:
            video2exp_embed = torch.zeros_like(video2exp_embed)

        text_audio_exp_embed = torch.cat([text2emo_embed, audio2emo_embed, video2exp_embed], dim=-1)

        text_audio_exp_embed = self.text_audio_video_mlp(text_audio_exp_embed)
        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析

        return text_audio_exp_embed
 