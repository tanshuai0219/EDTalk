import torch
import torch.nn.functional as F
from torch import nn

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out

class Audio2Lip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        #### load the pre-trained audio_encoder 
        #self.audio_encoder = self.audio_encoder.to(device)  
        # '''
        # wav2lip_state_dict = torch.load('/data/ts/code/comparison/Wav2Lip/checkpoints/wav2lip.pth')['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         # print('init:', k)
        #         state_dict[k.replace('module.audio_encoder.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)
        # '''

        self.mapping = nn.Linear(512, 20)
        nn.init.constant_(self.mapping.bias, 0.)

    def forward(self, x, batch_size, T): # #bs T 64
        x = self.audio_encoder(x).view(x.size(0), -1)
        
        y = self.mapping(x) 
        out = y.reshape(batch_size, T, -1) #+ ref # resudial
        return out


class Audio2Lip_nobank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        #### load the pre-trained audio_encoder 
        #self.audio_encoder = self.audio_encoder.to(device)  
        # '''
        wav2lip_state_dict = torch.load('/data/ts/code/comparison/Wav2Lip/checkpoints/wav2lip.pth')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                # print('init:', k)
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)
        # '''

        self.mapping = nn.Linear(512, 512)
        nn.init.constant_(self.mapping.bias, 0.)

    def forward(self, x, batch_size, T): # #bs T 64
        x = self.audio_encoder(x).view(x.size(0), -1)
        
        y = self.mapping(x) 
        out = y.reshape(batch_size, T, -1) #+ ref # resudial
        return out

if __name__ == '__main__':
    audio_encoder = Audio2Lip()
    audiox = torch.randn([20, 1,80,16])#mel_input.view(-1, 1, 80, 16)
    y = audio_encoder(audiox, 4,5)
    print(y.shape)
