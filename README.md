# Rectified Point Flow: Generic Point Cloud Pose Estimation

 [![ProjectPage](https://img.shields.io/badge/Project_Page-RPF-blue)](https://rectified-pointflow.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.05282-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2506.05282) [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/gradient-spaces) [![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)


 Official implementation of our paper: "Rectified Point Flow: Generic Point Cloud Pose Estimation".

[Tao Sun](https://taosun.io/) *<sup>,1</sup>,
[Liyuan Zhu](https://www.zhuliyuan.net/) *<sup>,1</sup>,
[Shengyu Huang](https://shengyuh.github.io/)<sup>2</sup>,
[Shuran Song](https://shurans.github.io/)<sup>1</sup>,
[Iro Armeni](https://ir0.github.io/)<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>NVIDIA Research | * denotes equal contribution

We introduce Rectified Point Flow, a unified parameterization that formulates pairwise point cloud registration and multi-part shape assembly as a single conditional generative problem. Given unposed point clouds, our method learns a continuous point-wise velocity field that transports noisy points toward their target positions, from which part poses are recovered. In contrast to prior work that regresses part-wise poses with ad-hoc symmetry handling, our method intrinsically learns assembly symmetries without symmetry labels.

<p align="center">
  <a href="">
    <img src="https://rectified-pointflow.github.io/images/overview_flow_asm.png" width="100%">
  </a>
</p>


We plan to release the code in June 2025. Thank you for your patience.


## BiBTeX
```bibtex
@inproceedings{sun2025_rpf,
      author = {Sun, Tao and Zhu, Liyuan and Huang, Shengyu and Song, Shuran and Armeni, Iro},
      title = {Rectified Point Flow: Generic Point Cloud Pose Estimation},
      booktitle = {arxiv preprint arXiv:2506.05282},
      year = {2025},
    }
```
