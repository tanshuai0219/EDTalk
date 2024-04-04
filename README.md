# EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis
The official repository of the paper [EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](https://arxiv.org/abs/2404.01647)

<p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2404.01647">Paper</a>
    | 
    <a href="https://tanshuai0219.github.io/EDTalk/">Project Page</a>
    |
    <a href="https://github.com/tanshuai0219/EDTalk">Code</a> 
  </b>
</p> 

<!-- Colab notebook demonstration: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egq0_ZK5sJAAawShxC0y4JRZQuVS2X-Z?usp=sharing) -->

  <p align='center'>  
    <img src='assets/image/teaser.svg' width='1000'/>
  </p>

Given an identity source, <strong>EDTalk</strong> synthesizes talking face videos characterized by mouth shapes, head poses, and expressions consistent with mouth GT, pose source and expression source. These facial dynamics can also be inferred directly from driven audio. Importantly, <strong>EDTalk</strong> demonstrates superior efficiency in disentanglement training compared to other methods.



## TODO
- [x] **Release Arxiv paper.**
- [ ] **Release code. (Once the paper is accepted)**
- [ ] **Release Pre-trained Model. (Once the paper is accepted)**



<!-- ## Citation	

```
@InProceedings{peng2023synctalk,
  title     = {SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis}, 
  author    = {Ziqiao Peng and Wentao Hu and Yue Shi and Xiangyu Zhu and Xiaomei Zhang and Jun He and Hongyan Liu and Zhaoxin Fan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
}
``` -->

## Acknowledgement
Some code are borrowed from following projects:
* [LIA](https://github.com/wyhsirius/LIA)
* [DPE](https://github.com/OpenTalker/DPE)
* [EAT](https://github.com/yuangan/EAT_code)
* [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference)
* [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
* [FOMM video preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing)

Some figures in the paper is inspired by:
* [PD-FGC](https://arxiv.org/abs/2211.14506)
* [DreamTalk](https://arxiv.org/abs/2312.09767)

The README.md template is borrowed from [SyncTalk](https://github.com/ziqiaopeng/SyncTalk)


Thanks for these great projects.
