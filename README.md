<!-- # EDTalk -->


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
<i><strong><a href='https://eccv2024.ecva.net/' target='_blank'>ECCV 2024 Oral</a></strong></i>
</p>



<div align="center">
  <a href="https://tanshuai0219.github.io/EDTalk/"><img src="https://img.shields.io/badge/project-EDTalk-red"></a> &ensp;
  <a href="https://arxiv.org/abs/2404.01647"><img src="https://img.shields.io/badge/Arxiv-EDTalk-blue"></a> &ensp;
  <a href="https://github.com/tanshuai0219/EDTalk"><img src="https://img.shields.io/github/stars/tanshuai0219/EDTalk?style=social"></a> &ensp;

  <!-- [![GitHub Stars](https://img.shields.io/github/stars/yuangan/EAT_code?style=social)](https://github.com/yuangan/EAT_code) -->
<!--   <a href="https://arxiv.org/abs/2404.01647"><img src="https://img.shields.io/badge/OpenXlab-EDTalk-grenn"></a> &ensp; -->

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
- 2025.09.28 - 🎉 Our new paper [FixTalk](https://arxiv.org/pdf/2507.01390) is selected as best paper finalist by ICCV 2025. FixTalk is built on EDTalk, we look forward to seeing more innovations based on EDTalk~
- 2025.07.24 - 🎉 Our new paper [FixTalk](https://arxiv.org/pdf/2507.01390) is accepted as an oral presentation by ICCV 2025. FixTalk is built on EDTalk, we look forward to seeing more innovations based on EDTalk~
- 2024.12.30 - 🎉 Another talk about EDTalk at [https://byuih.xetlk.com/sl/40yc8X](https://wqpoq.xetlk.com/sl/1qGlJZ) (But it's in Chinese, too~).
- 2024.10.02 - 🎉 I give an oral presentation about EDTalk at ECCV Milan.
- 2024.09.20 - If you want to compare EDTalk in neutral talking head generation without emotional expression for the coming **ICLR and CVPR**, [demo_lip_pose.py](https://github.com/tanshuai0219/EDTalk/blob/main/demo_lip_pose.py) is recommended.
- 2024.08.28 - 🎉 I give a presentation about EDTalk at https://byuih.xetlk.com/sl/40yc8X (But it's in Chinese~).
- 2024.08.12 - 🎉 Our paper is selected as an oral presentation.
- 2024.08.09 - 💻 Add the training code for fine-tuning on a specific person, and we take Obama as example.
- 2024.08.06 - 🙏 We hope more people can get involved, and we will promptly handle pull requests. Currently, there are still some tasks that need assistance, such as creating a colab notebook, improved web UI, and translation work, among others.
- 2024.08.04 - 🎉 Add gradio interface.
- 2024.07.31 - 💻 Add optional face super-resolution.
- 2024.07.19 - 💻 Release data preprocess codes and partial training codes (fine-tuning LIA & Mouth-Pose Decouple & Audio2Mouth). But I'm busy now and don't have enough time to clean up all the codes, but I think the current codes can be a useful reference if ones want to reproduce EDTalk or other. If you run into any problems, feel free to propose an issue!
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

- python packages for Windows
```
pip install -r requirements_windows.txt
```

  Thanks to [nitinmukesh](https://github.com/nitinmukesh) for providing a [Windows 11 installation tutorial](https://www.youtube.com/watch?v=KLnMyspiOMk) and welcome to follow his channel!

- Launch gradio interface (Thank the contributor: [newgenai79](https://github.com/newgenai79)!)
```
python webui_emotions.py
```

<div align="center">
  <img src="assets/image/gradio.png" width="800" ></img>
  <br>
</div>



## 🎬 Quick Start

Download the [checkpoints](https://drive.google.com/file/d/1EKJXpq5gwFaRfkiAs6YUZ6YEiQ-8X3H3/view?usp=drive_link)/[huggingface link](https://huggingface.co/tanhshuai0219/EDTalk/tree/main) and put them into ./ckpts.

[中文用户] 可以通过这个[链接](https://openxlab.org.cn/models/detail/tanshuai0219/EDTalk/tree/main)下载权重。


### **EDTalk-A:lip+pose+exp**: Run the demo in audio-driven setting (EDTalk-A):
#### For user-friendliness, we extracted the weights of eight common sentiments in the expression base. one can directly specify the sentiment to generate emotional talking face videos (recommended)
  ```
  python demo_EDTalk_A_using_predefined_exp_weights.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_type type/of/expression --save_path path/to/save

  # example:
  python demo_EDTalk_A_using_predefined_exp_weights.py --source_path res/results_by_facesr/demo_EDTalk_A.png --audio_driving_path test_data/mouth_source.wav --pose_driving_path test_data/pose_source1.mp4 --exp_type angry --save_path res/demo_EDTalk_A_using_weights.mp4
  ```
  ****

|  |           |         |
  |------------|--------------------------|---------------------------|
  |<video controls loop src="https://github.com/user-attachments/assets/09ff9885-073b-4750-bec5-f1574126d6eb" muted="false"></video> | <video controls loop src="https://github.com/user-attachments/assets/f5c7ff63-66dd-45cb-9e2d-7808fbb0fbaf" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/3dbad54f-bd23-4e28-83c7-d941e58d506a" muted="false"></video> |
  |<video controls loop src="https://github.com/user-attachments/assets/9a1423f6-3b5d-4cfc-8658-7b9d2fd348d2" muted="false"></video> | <video controls loop src="https://github.com/user-attachments/assets/c29148f6-6083-4ef7-8b76-3340ad32a832" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/0c8fed0f-3267-4507-9c43-db7464d65abf" muted="false"></video> |

  
#### Or you can input an expression reference (image/video) to indicate expression.

  ```
  python demo_EDTalk_A.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_driving_path path/to/expression --save_path path/to/save

  # example:
  python demo_EDTalk_A.py --source_path res/results_by_facesr/demo_EDTalk_A.png --audio_driving_path test_data/mouth_source.wav --pose_driving_path test_data/pose_source1.mp4 --exp_driving_path test_data/expression_source.mp4 --save_path res/demo_EDTalk_A.mp4

  ```
  The result will be stored in save_path.

  **Source_path and videos used must be first cropped using scripts [crop_image2.py](data_preprocess/crop_image2.py) (download [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and put it in ./data_preprocess dir) and [crop_video.py](data_preprocess/crop_video.py). Make sure the every video' frame rate must be 25 fps**

  You can also use [crop_image.py](data_preprocess/crop_image.py) to crop the image, but [increase_ratio](https://github.com/tanshuai0219/EDTalk/blob/928fe3de7cf74b6a0e7db4ec90d59c85d79b8bc1/data_preprocess/crop_image.py#L76) has to be carefully set and tried several times to get the optimal result.
  <!-- For images where faces only make up a small portion of the image, we recommend using the [crop_image2.py](data_preprocess/crop_image2.py) to crop image. -->

****
### **EDTalk-A:lip+pose without exp**: If you don't want to change the expression of the identity source, please download the [EDTalk_lip_pose.pt](https://drive.google.com/file/d/1XkCWeph0LvQfpWb2mO4YhfUVE3qay71Z/view?usp=sharing) and put it into ./ckpts.

#### If you only want to change the lip motion of the identity source, run
  ```
   python demo_lip_pose.py --fix_pose --source_path path/to/image --audio_driving_path path/to/audio --save_path path/to/save
   # example:
   python demo_lip_pose.py --fix_pose --source_path test_data/identity_source.jpg --audio_driving_path test_data/mouth_source.wav --save_path res/demo_lip_pose_fix_pose.mp4
  ```
****
#### Or you can additionally control the head poses on top of the above via pose_driving_path
  ```
   python demo_lip_pose.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --save_path path/to/save

   # example:
   python demo_lip_pose.py --source_path test_data/identity_source.jpg --audio_driving_path test_data/mouth_source.wav --pose_driving_path test_data/pose_source1.mp4 --save_path res/demo_lip_pose_fix_pose.mp4

  ```

| Source Img | EDTalk        | EDTalk + liveprotrait           |
|------------|--------------------------|---------------------------|
|<img src="https://github.com/user-attachments/assets/1620d456-7bbf-436b-8bad-fdcd247e9f26" width="250" ></img> | <video controls loop src="https://github.com/user-attachments/assets/36ae9b6d-fc96-476a-8e63-8fe318b32782" muted="false"></video> |  |
|<img src="https://github.com/user-attachments/assets/22fd0a6a-dc00-4719-9bc8-9778fd5b0e79" width="250" ></img> | <video controls loop src="https://github.com/user-attachments/assets/70c27d4b-dd06-4ae1-81ad-7e4795fce541" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/5cfb1933-ec7c-48a6-8343-507f5fd4a090" muted="false"></video> |

#### And control the lip motion via a driven video.
  ```
   python demo_lip_pose_V.py --source_path path/to/image --audio_driving_path path/to/audio --lip_driving_path path/to/mouth --pose_driving_path path/to/pose --save_path path/to/save

  # example:
   python demo_lip_pose_V.py --source_path res/results_by_facesr/demo_lip_pose5.png --audio_driving_path test_data/mouth_source.wav --lip_driving_path test_data/mouth_source.mp4 --pose_driving_path test_data/pose_source1.mp4 --save_path demo_lip_pose_V.mp4

  ```
| Source Img | demo_lip_pose_V Results           | + FaceSR           |
|------------|--------------------------|---------------------------|
|<img src="test_data/identity_source.jpg" width="250" ></img> | <video controls loop src="https://github.com/user-attachments/assets/912097cf-ce92-42ca-960b-c4e0906cb0b0" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/c4e1a81c-76c1-462a-b671-9c82e37e14ad" muted="false"></video> |
|<img src="test_data/leijun.png" width="250" ></img> | <video controls loop src="https://github.com/user-attachments/assets/4e630594-1dd2-47fb-b367-6be7a700c769" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/f1a0b477-a120-47a5-b925-00af4ff09781" muted="false"></video> |

#### Change the lip motion of a source video, run:
  ```
   python demo_change_a_video_lip.py --source_path path/to/video --audio_driving_path path/to/audio --save_path path/to/save

   # example
   python demo_change_a_video_lip.py --source_path test_data/pose_source1.mp4 --audio_driving_path test_data/mouth_source.wav --save_path res/demo_change_a_video_lip.mp4

  ```
| Source Img | results #1           | results #2          |
|------------|--------------------------|---------------------------|
|<video controls loop src="https://github.com/user-attachments/assets/f940a507-d28c-4cc9-abda-af82c6bbf596" muted="false"></video> | <video controls loop src="https://github.com/user-attachments/assets/d199732f-66ad-4182-9df1-0e4416ec8a51" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/328d2b9d-8e98-4814-9d6f-195dddfd80f7" muted="false"></video> |



****
### Run the demo in video-driven setting (EDTalk-V):
  ```
  python demo_EDTalk_V.py --source_path path/to/image --lip_driving_path path/to/lip --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_driving_path path/to/expression --save_path path/to/save

  # example:
  python demo_EDTalk_V.py --source_path test_data/identity_source.jpg --lip_driving_path test_data/mouth_source.mp4 --audio_driving_path test_data/mouth_source.wav --pose_driving_path test_data/pose_source1.mp4 --exp_driving_path test_data/expression_source.mp4 --save_path res/demo_EDTalk_V.mp4

  ```
  The result will be stored in save_path.


## Face Super-resolution (Optional)

☺️🙏 Thanks to [Tao Liu](https://github.com/liutaocode) for the proposal~

The purpose is to upscale the resolution from 256 to 512 and address the issue of blurry rendering.

Please install addtional environment here:

```
pip install facexlib
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
pip install gfpgan
```



Then enable the option `--face_sr` in your scripts. The first time will download the weights of gfpgan (you can optionally first download [gfpgan ckpts](https://drive.google.com/file/d/1SEWp_lnvxTHI1EIzurbNGYbPABmxih8A/view?usp=sharing) and put them in gfpgan/weights dir).

Here are some examples:

  ```

  python demo_lip_pose.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --save_path path/to/save --face_sr

  python demo_EDTalk_V.py --source_path path/to/image --lip_driving_path path/to/lip --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_driving_path path/to/expression --save_path path/to/save --face_sr

  python demo_EDTalk_A_using_predefined_exp_weights.py --source_path path/to/image --audio_driving_path path/to/audio --pose_driving_path path/to/pose --exp_type type/of/expression --save_path path/to/save --face_sr
  ```

<!-- **Note:** Due to the limitations of markdown, we downsampled the results after facesr, which may be detrimental to video quality and smoothness, see the [results_by_facesr](res/results_by_facesr) for detailed results.

| Source Img | EDTalk Results           | EDTalk + FaceSR           |
|------------|--------------------------|---------------------------|
|<img src="res/results_by_facesr/demo_lip_pose5.png" width="200" ></img> | <img src="res/results_by_facesr/gif/demo_lip_pose5.gif" width="200" ></img> |  <img src="res/results_by_facesr/gif/demo_lip_pose5_512.gif" width="200" ></img> |
|<img src="res/results_by_facesr/demo_EDTalk_A.png" width="200" ></img> | <img src="res/results_by_facesr/gif/demo_EDTalk_A.gif" width="200" ></img> |  <img src="res/results_by_facesr/gif/demo_EDTalk_A_512.gif" width="200" ></img>      |
|<img src="res/results_by_facesr/RD_Radio51_000.png" width="200" ></img> | <img src="res/results_by_facesr/gif/RD_Radio51_000.gif" width="200" ></img>  |   <img src="res/results_by_facesr/gif/RD_Radio51_000_512.gif" width="200" ></img>     | -->


| Source Img | EDTalk Results           | EDTalk + FaceSR           |
|------------|--------------------------|---------------------------|
|<img src="res/results_by_facesr/demo_lip_pose5.png" width="250" ></img> |<video controls loop src="https://github.com/user-attachments/assets/f450414f-e272-49eb-a39e-0ffcb9269470" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/6ad42d0b-6c3d-498b-b16f-0bb0fc7699b7" muted="false"></video> |
|<img src="res/results_by_facesr/demo_EDTalk_A.png" width="250" ></img> | <video controls loop src="https://github.com/user-attachments/assets/8ca59ada-507c-4d4e-a126-0e806582b4b6" muted="false"></video> |  <video controls loop src="https://github.com/user-attachments/assets/bccea19d-513c-4c22-8c49-4aac7c7d49d0" muted="false"></video>      |
|<img src="res/results_by_facesr/RD_Radio51_000.png" width="250" ></img> | <video controls loop src="https://github.com/user-attachments/assets/b75f5a6c-0d38-4dc2-bbfa-330290f098ba" muted="false"></video>  |   <video controls loop src="https://github.com/user-attachments/assets/644100c6-608e-4266-8b94-6b61880dddbe" muted="false"></video>     |


## 🎬 Fine tune on a specific person 
There are a few issues currently, I'll be checking them carefully. Please be patient!
**Note**: We take Obama and the path in my computer (/data/ts/xxxxxx) as example and you should replace it with your own path:

- Download the Obama data from [AD-Nerf](https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4) and put it in '/data/ts/datasets/person_specific_dataset/AD-NeRF/video/Obama.mp4'

- Crop video and resample as 25 fps:
    ```bash
    python data_preprocess/crop_video.py --inp /data/ts/datasets/person_specific_dataset/AD-NeRF/video/Obama.mp4  --outp /data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop/Obama.mp4 
    ```
- Save video as frames:
    ```bash
    ffmpeg -i /data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop/Obama.mp4 -r 25 -f image2 /data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop_frame/Obama/%4d.png
    ```

- Start training:
    ```bash
    python train_fine_tune.py --datapath /data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop_frame/Obama --only_fine_tune_dec
    ```
  Change datapath as your own data. only_fine_tune_dec means only training dec module. In my experience, training only dec can help with image quality, so we recommend it. You also can set it as False, and it means to fune tune full model. You should go through the saved samples (at exp_path/exp_name/checkpoint and in my case, at: /data/ts/checkpoints/EDTalk/fine_tune/Obama/checkpoint) frequently to find the optimal model in time.


| Step #0 | Step #100          | Step #200          |
|------------|--------------------------|---------------------------|
|<img src="fine_tune/examples/step_00000.jpg" width="250" ></img> | <img src="fine_tune/examples/step_00200.jpg" width="250" ></img> |  <img src="fine_tune/examples/step_00400.jpg" width="250" ></img> |

    First line is source image, second line is driving image, and third line is generated results.

##  🎬 Data Preprocess for Training
<details> <summary> Data Preprocess for Training </summary>
**Note**: The functions provided are available, but one should adjust the way they are called, e.g. by modifying the path to the data. If you run into any problems, feel free to leave your problems!
- Download the MEAD and HDTF dataset:
1) **MEAD**. [download link](https://wywu.github.io/projects/MEAD/MEAD.html). 


    We only use *Front* videos and extract audios and orgnize the data as follows:

    ```text
    /dir_path/MEAD_front/
    |-- Original_video
    |   |-- M003#angry#level_1#001.mp4
    |   |-- M003#angry#level_1#002.mp4
    |   |-- ...
    |-- audio
    |   |-- M003#angry#level_1#001.wav
    |   |-- M003#angry#level_1#002.wav
    |   |-- ...
    ```


2) **HDTF**. [download link](https://github.com/MRzzm/HDTF).

    We orgnize the data as follows:

    ```text
    /dir_path/HDTF/
    |-- audios
    |   |-- RD_Radio1_000.wav
    |   |-- RD_Radio2_000.wav
    |   |-- ...
    |-- original_videos
    |   |-- RD_Radio1_000.mp4
    |   |-- RD_Radio2_000.mp4
    |   |-- ...
    ```

- Crop videos in training datasets:
    ```bash
    python data_preprocess/data_preprocess_for_train/crop_video_MEAD.py
    python data_preprocess/data_preprocess_for_train/crop_video_HDTF.py
    ```
- Split video: Since the video in HDTF is too long, we split both the video and the corresponding audio into 5s segments:
    ```bash
    python data_preprocess/data_preprocess_for_train/split_HDTF_video.py
    ```

    ```bash
    python data_preprocess/data_preprocess_for_train/split_HDTF_audio.py
    ```

- We save the video frames in a lmdb file to improve I/O efficiency:
    ```bash
    python data_preprocess/data_preprocess_for_train/prepare_lmdb.py
    ```

- Extract mel feature from audio:
    ```bash
    python data_preprocess/data_preprocess_for_train/get_mel.py
    ```
- Extract landmarks from cropped videos:
    ```bash
    python data_preprocess/data_preprocess_for_train/extract_lmdk.py
    ```
- Extract bboxs from cropped videos for lip discriminator using [extract_bbox.py](https://github.com/yuangan/EAT_code/blob/main/preprocess/extract_bbox.py) and we give an unclean example using lmdb like :
    ```bash
    python data_preprocess/data_preprocess_for_train/extract_bbox.py
    ```

- After the preprocessing, the data should be orgnized as follows:
    ```text
    /dir_path/MEAD_front/
    |-- Original_video
    |   |-- M003#angry#level_1#001.mp4
    |   |-- M003#angry#level_1#002.mp4
    |   |-- ...
    |-- video
    |   |-- M003#angry#level_1#001.mp4
    |   |-- M003#angry#level_1#002.mp4
    |   |-- ...
    |-- audio
    |   |-- M003#angry#level_1#001.wav
    |   |-- M003#angry#level_1#002.wav
    |   |-- ...
    |-- bbox
    |   |-- M003#angry#level_1#001.npy
    |   |-- M003#angry#level_1#002.npy
    |   |-- ...
    |-- landmark
    |   |-- M003#angry#level_1#001.npy
    |   |-- M003#angry#level_1#002.npy
    |   |-- ...
    |-- mel
    |   |-- M003#angry#level_1#001.npy
    |   |-- M003#angry#level_1#002.npy
    |   |-- ...

    /dir_path/HDTF/
    |-- split_5s_video
    |   |-- RD_Radio1_000#1.mp4
    |   |-- RD_Radio1_000#2.mp4
    |   |-- ...
    |-- split_5s_audio
    |   |-- RD_Radio1_000#1.wav
    |   |-- RD_Radio1_000#2.wav
    |   |-- ...
    |-- bbox
    |   |-- RD_Radio1_000#1.npy
    |   |-- RD_Radio1_000#2.npy
    |   |-- ...
    |-- landmark
    |   |-- RD_Radio1_000#1.npy
    |   |-- RD_Radio1_000#2.npy
    |   |-- ...
    |-- mel
    |   |-- RD_Radio1_000#1.npy
    |   |-- RD_Radio1_000#2.npy
    |   |-- ...

    ```
  </details>
## 🎬 Start Training
<details> <summary> Start Training </summary>
- Pretrain Encoder $E$ and Generator $G$:

    - Please refer to [LIA](https://github.com/wyhsirius/LIA) to train from scratch.
    - (Optional) If you want to accelerate convergence speed, you can download the pre-trained model of [LIA](https://github.com/wyhsirius/LIA).
    - we provide training code to fine-tune the model on MEAD and HDTF dataset:
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 train/train_E_G.py
    ```

- Train Mouth-Pose Decouple module:
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 12344 train/train_Mouth_Pose_decouple.py
    ```
<!-- - Train Expression Decouple module:
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 12344 train/train_Expression_decouple.py
    ``` -->
- Train Audio2Mouth module:
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 12344 train/train_audio2mouth.py
    ```
   </details>


## 🙏 Thanks to all contributors for their efforts

We hope more people can get involved, and we will promptly handle pull requests. Currently, there are still some tasks that need assistance, such as creating a colab notebook, web UI, and translation work, among others.

[![contributors](https://contrib.rocks/image?repo=tanshuai0219/EDTalk)](https://github.com/tanshuai0219/EDTalk/graphs/contributors)






## 👨‍👩‍👧‍👦 Other Talking head papers:

[ICCV 23] [EMMN: Emotional Motion Memory Network for Audio-driven Emotional Talking Face Generation](http://openaccess.thecvf.com/content/ICCV2023/html/Tan_EMMN_Emotional_Motion_Memory_Network_for_Audio-driven_Emotional_Talking_Face_ICCV_2023_paper.html)

[AAAI 24] [Style2Talker: High-Resolution Talking Head Generation with Emotion Style and Art Style](https://ojs.aaai.org/index.php/AAAI/article/view/28313)

[AAAI 24] [Say Anything with Any Style](https://ojs.aaai.org/index.php/AAAI/article/view/28314)

[CVPR 24] [FlowVQTalker: High-Quality Emotional Talking Face Generation through Normalizing Flow and Quantization](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_FlowVQTalker_High-Quality_Emotional_Talking_Face_Generation_through_Normalizing_Flow_and_CVPR_2024_paper.html)

[ICCV 25] [FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases](https://arxiv.org/pdf/2507.01390)



## 🎓 Citation

```
@inproceedings{tan2025edtalk,
  title={Edtalk: Efficient disentanglement for emotional talking head synthesis},
  author={Tan, Shuai and Ji, Bin and Bi, Mengxiao and Pan, Ye},
  booktitle={European Conference on Computer Vision},
  pages={398--416},
  year={2025},
  organization={Springer}
}

@inproceedings{tan2023emmn,
  title={Emmn: Emotional motion memory network for audio-driven emotional talking face generation},
  author={Tan, Shuai and Ji, Bin and Pan, Ye},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22146--22156},
  year={2023}
}

@inproceedings{tan2025fixtalk,
  title={FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases},
  author={Tan, Shuai and Gong, Bill and Ji, Bin and Pan, Ye},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}

@inproceedings{tan2024say,
  title={Say anything with any style},
  author={Tan, Shuai and Ji, Bin and Ding, Yu and Pan, Ye},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={5088--5096},
  year={2024}
}

@inproceedings{tan2024style2talker,
  title={Style2Talker: High-Resolution Talking Head Generation with Emotion Style and Art Style},
  author={Tan, Shuai and Ji, Bin and Pan, Ye},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={5079--5087},
  year={2024}
}

@inproceedings{tan2024flowvqtalker,
  title={FlowVQTalker: High-Quality Emotional Talking Face Generation through Normalizing Flow and Quantization},
  author={Tan, Shuai and Ji, Bin and Pan, Ye},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26317--26327},
  year={2024}
}

```

## 🙏 Acknowledgement
Some code are borrowed from following projects:
* [LIA](https://github.com/wyhsirius/LIA)
* [EAT](https://github.com/yuangan/EAT_code)
* [DPE](https://github.com/OpenTalker/DPE)
* [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference)
* [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
* [FOMM video preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing)

Some figures in the paper is inspired by:
* [PD-FGC](https://arxiv.org/abs/2211.14506)
* [DreamTalk](https://arxiv.org/abs/2312.09767)



Thanks for these great projects.
