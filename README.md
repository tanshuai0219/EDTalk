<!-- # ResMaster -->


### <div align="center">🚀 EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis</div> 


<p align="center">
  <a href="https://scholar.google.com.hk/citations?user=9KjKwDwAAAAJ&hl=en">Shuai Tan</a><sup>1</sup>,
  <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=uZeBvd8AAAAJ">Bin Ji</a><sup>1</sup>, 
  <a href="">Mengxiao Bi</a><sup>2</sup>, 
  <a href="">Ye Pan</a><sup>1</sup>, 
  <br><br>
  <sup>1</sup>Shanghai Jiao Tong University<br>
  <sup>2</sup>NetEase Fuxi AI Lab<br>
  <br>
<i><strong><a href='https://eccv2024.ecva.net/' target='_blank'>ECCV 2024</a></strong></i>
<br>
</p>



<br>

<div align="center">
  <a href="https://tanshuai0219.github.io/EDTalk/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2404.01647"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;

</div>

<div align="center">
  <img src="assets/image/teaser.svg" width="900" ></img>
  <br>
</div>
<br>


## 🎏 Abstract
Achieving disentangled control over multiple facial motions and accommodating diverse input modalities greatly enhances the application and entertainment of the talking head generation. This necessitates a deep exploration of the decoupling space for facial features, ensuring that they <strong>a)</strong> operate independently without mutual interference and <strong>b)</strong> can be preserved to share with different modal inputs—both aspects often neglected in existing methods. To address this gap, this paper proposes a novel <strong>E</strong>fficient <strong>D</strong>isentanglement framework for <strong>Talk</strong>ing head generation (<strong>EDTalk</strong>). Our framework enables individual manipulation of mouth shape, head pose, and emotional expression, conditioned on both video and audio inputs. Specifically, we employ three <strong>lightweight</strong> modules to decompose the facial dynamics into three distinct latent spaces representing mouth, pose, and expression, respectively. Each space is characterized by a set of learnable bases whose linear combinations define specific motions. To ensure independence and accelerate training, we enforce orthogonality among bases and devise an <strong>efficient</strong> training strategy to allocate motion responsibilities to each space without relying on external knowledge. The learned bases are then stored in corresponding banks, enabling shared visual priors with audio input. Furthermore, considering the properties of each space, we propose Audio-to-Motion module for audio-driven talking head synthesis. Experiments are conducted to demonstrate the effectiveness of EDTalk.
## 💻 Overview
<div align="center">
  <img src="assets/image/EDTalk.png" width="800" ></img>
  <br>
</div>
<br>
<!-- Illustration of our proposed EDTalk. (a) EDTalk framework. Given an identity source $ I^i $ and various driving images $I^*$ ($ * \in \{m,p,e\} $) for controlling corresponding facial components, EDTalk animates the identity image $ I^i $ to mimic the mouth shape, head pose, and expression of $ I^m $, $ I^p $, and $ I^e $ with the assistance of three Component-aware Latent Navigation modules: MLN, PLN, and ELN.  (b) Efficient Disentanglement. The disentanglement process consists of two parts: Mouth-Pose decouple and Expression Decouple. For the former, we introduce the cross-reconstruction training strategy aimed at separating mouth shape and head pose. For the latter, we achieve expression disentanglement using self-reconstruction complementary learning. -->


## 🔥 Update

- 2024.07.01 - 💻 The inference code and pretrained models are available.
- 2024.07.01 - 🎉 Our paper is accepted by ECCV 2024.
- 2024.04.02 - 🛳️ This repo is released.


## 📅 TODO

- [ ] **Release training code.**
- [x] **Release inference code.**
- [x] **Release pre-trained models.**
- [x] **Release Arxiv paper.**


## 🎮 Installation
We train and test based on Python 3.8 and Pytorch. To install the dependencies run:
```bash
git clone https://github.com/tanshuai0219/EDTalk.git
cd EDTalk
```

### Install dependency
```
conda create -n EDTalk python=3.8
conda activate EDTalk
```

- python packages
```
pip install -r requirements.txt
```




## 🎬 Quick Start

Download the [checkpoints](https://drive.google.com/file/d/1EKJXpq5gwFaRfkiAs6YUZ6YEiQ-8X3H3/view?usp=drive_link) and put them into ./ckpts.


### Run the demo in audio-driven setting (EDTalk-A):
#### For user-friendliness, we extracted the weights of eight common sentiments in the expression base. one can directly specify the sentiment to generate emotional talking face videos (recommended)
  ```
  python demo_EDTalk_A.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_driving_path path/to/expression --save_path path/to/save
  ```
#### Or you can input an expression reference (image/video) to indicate expression.

  ```
  python demo_EDTalk_A.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_driving_path path/to/expression --save_path path/to/save
  ```
  The result will be stored in save_path.

  **Source_path and videos used must be first cropped using scripts [crop_image.py](data_preprocess/crop_image.py) and [crop_video.py](data_preprocess/crop_video.py)**

### Run the demo in video-driven setting (EDTalk-V):
  ```
  python demo_EDTalk_V.py --source_path path/to/image --lip_driving_path path/to/lip --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_driving_path path/to/expression --save_path path/to/save
  ```
  The result will be stored in save_path.

## 🎓 Citation

```
@article{tan2024edtalk,
  title={EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis},
  author={Tan, Shuai and Ji, Bin and Bi, Mengxiao and Pan, Ye},
  journal={arXiv preprint arXiv:2404.01647},
  year={2024}
}
```

## 🙏 Acknowledgement
Some code are borrowed from following projects:
* [LIA](https://github.com/wyhsirius/LIA)
* [DPE](https://github.com/OpenTalker/DPE)
* [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference)
* [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
* [FOMM video preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing)

Some figures in the paper is inspired by:
* [PD-FGC](https://arxiv.org/abs/2211.14506)
* [DreamTalk](https://arxiv.org/abs/2312.09767)

The README.md template is borrowed from [SyncTalk](https://github.com/ziqiaopeng/SyncTalk)


Thanks for these great projects.