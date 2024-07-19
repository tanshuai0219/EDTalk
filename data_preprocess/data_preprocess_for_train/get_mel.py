# -*- coding: utf-8 -*-

import numpy as np

from demo_EDTalk_A import *

audio_file = 'test.wav'
save_path = 'test.mel'
source_audio_feature, source_nums = get_mel(audio_file)
np.save(save_path, source_audio_feature)