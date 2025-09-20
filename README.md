<p align="center">
  <h1 align="center">VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction</h1>
  <p align="center">
    <a href="https://lhmd.top">Weijie Wang*</a>
    ·
    <a href="https://github.com/Alresd">YeQing Chen*</a>
    ·
    <a href="https://steve-zeyu-zhang.github.io">Zeyu Zhang</a>
    ·
    <a href="https://liuhengyu321.github.io">Hengyu Liu</a>
    ·
    <a href="https://wang-haoxiao.github.io">Haoxiao Wang</a>
    ·
    <a href="https://scholar.google.com/citations?user=4HaLG0oAAAAJ">Zhiyuan Feng</a>
    ·
    <a href="https://scholar.google.com/citations?user=TE9stNgAAAAJ">Wenkang Qin</a>
    ·
    <a href="http://www.zhengzhu.net/">Zheng Zhu</a>
    ·
    <a href="https://donydchen.github.io">Donny Y. Chen</a>
    ·
    <a href="https://bohanzhuang.github.io">Bohan Zhuang</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2505.23734">Paper</a> | <a href="https://lhmd.top/volsplat">Project Page</a> | <a href="https://github.com/ziplab/VolSplat">Code</a> | <a href="https://huggingface.co/lhmd/VolSplat">Models</a> </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="https://lhmd.top/volsplat/assets/teaser_horizontal.jpg" alt="Logo" width="100%">
  </a>
</p>
Pixel-aligned feed-forward 3DGS methods suffer from two primary limitations: 1) 2D feature matching struggles to effectively resolve the multi-view alignment problem, and 2) the Gaussian density is constrained and cannot be adaptively controlled according to scene complexity. We propose VolSplat, a method that directly regresses Gaussians from 3D features based on a voxel-aligned prediction strategy. This approach achieves adaptive control over scene complexity and resolves the multi-view alignment challenge.

## Method
<p align="center">
  <a href="">
    <img src="https://lhmd.top/volsplat/assets/pipeline.jpg" alt="Logo" width="100%">
  </a>
</p>
<strong>Overview of VolSplat</strong>. Given multi-view images as input, we first extract 2D features for each image using a Transformer-based network and construct per-view cost volumes with plane sweeping. Depth Prediction Module then estimates a depth map for each view, which is used to unproject the 2D features into 3D space to form a voxel feature grid. Subsequently, we employ a sparse 3D decoder to refine these features in 3D space and predict the parameters of a 3D Gaussian for each occupied voxel. Finally, novel views are rendered from the predicted 3D Gaussians.


## TODOs
- [ ] Release Code.
- [ ] Release Model Checkpoints.

## Citation
If you find our work useful for your research, please consider citing us:

```bibtex
@article{wang2025volsplat,
  title={VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction},
  author={Wang, Weijie and Chen, YeQing and Zhang, Zeyu and Liu, Hengyu and Wang, Haoxiao and Feng, Zhiyuan and Qin, Wenkang and Zhu, Zheng and Chen, Donny Y. and Zhuang, Bohan},
  journal={},
  year={2025}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at wangweijie@zju.edu.cn.

## Acknowledgements

This project is developed with [DepthSplat](https://github.com/cvg/depthsplat). We thank the original authors for their excellent work.
