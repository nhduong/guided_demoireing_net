<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://nhduong.github.io/">
    <img src="dgu.png" alt="Logo" width="224" height="224">
  </a> -->

  <h2 align="center">Multiscale Guided Coarse-to-Fine Network for Screenshot Demoiréing</h2>

  <p align="center">
    <a href="mailto:duongnguyen@mme.dongguk.edu" target="_blank">Duong Hai Nguyen</a><sup>1</sup>,
    <a href="mailto:seholee@jbnu.ac.kr" target="_blank">Se-Ho Lee</a><sup>2</sup>, and 
    <a href="mailto:chullee@dongguk.edu" target="_blank">Chul Lee</a><sup>1</sup>
    <br>
    <sup>1</sup>Department of Multimedia Engineering, Dongguk University, South Korea<br>
    <sup>2</sup>Department of Information and Engineering, Jeonbuk National University, South Korea<br>
    IEEE Signal Processing Letters 2023
    <br>
    <br>
    <a href="https://nhduong.github.io/guided_demoireing_net">Project Page</a>
    ·
    <a href="https://doi.org/10.1109/LSP.2023.3296039">Paper</a>
  </p>
</p>

<br>
<br>
<br>

# Installation
1. Clone this repo:
```bash
git clone https://github.com/nhduong/guided_adaptive_demoireing.git
```

2. Install dependencies:
```bash
conda create -n <environment-name> --file requirements.txt
```

3. Download datasets

| Dataset | Download Link |
| :---: | :---: |
| LCDMoiré | [CodaLab](https://competitions.codalab.org/competitions/20165) |
| TIP2018 | [Google Drive](https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC) |
| FHDMi | [Google Drive](https://drive.google.com/drive/folders/1IJSeBXepXFpNAvL5OyZ2Y1yu4KPvDxN5) |
| UHDM | [Google Drive](https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e) |

# Testing
1. Download [pretrained models](https://drive.google.com/drive/folders/1PNC3Q8Iqh9Ksg9zilsidgAF1lWCe2wJN?usp=sharing) from Google Drive

2. Execute the following commands
```bash
# for LCDMoiré
CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
      --data_path "path_to/aim2019_demoireing_track1" \
      --data_name aim --train_dir "train" --test_dir "val" --moire_dir "moire" --clean_dir "clear" \
      --resume "path_to/aim/checkpoint.pth.tar"

# for TIP2018
CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
      --data_path "path_to/TIP2018_original" \
      --data_name tip18 --train_dir "trainData" --test_dir "testData" --moire_dir "source" --clean_dir "target" \
      --resume "path_to/tip18/checkpoint.pth.tar"

# for FHDMi
CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
      --data_path "path_to/FHDMi_complete" \
      --data_name fhdmi --train_dir "train" --test_dir "test" --moire_dir "source" --clean_dir "target" \
      --resume "path_to/fhdmi/checkpoint.pth.tar" --num_branches 4

# for UHDM
CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
        --data_path "path_to/UHDM_DATA" \
        --data_name uhdm --train_dir "train" --test_dir "test" --moire_dir "" --clean_dir "" \
        --resume "path_to/uhdm/checkpoint.pth.tar" --num_branches 4

```

# Training

1. Run the following commands

```bash
# for LCDMoiré
CUDA_VISIBLE_DEVICES="GPU_ID" nohup accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --dont_calc_mets_at_all --log2file \
    --data_path "path_to/aim2019_demoireing_track1" \
    --data_name aim --train_dir "train" --test_dir "val" --moire_dir "moire" --clean_dir "clear" \
    --batch_size 2 --T_0 50 --epochs 200 --init_weights &

# for TIP2018
CUDA_VISIBLE_DEVICES="GPU_ID" nohup accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --dont_calc_mets_at_all --log2file \
    --data_path "path_to/TIP2018_original" \
    --data_name tip18 --train_dir "trainData" --test_dir "testData" --moire_dir "source" --clean_dir "target" \
    --batch_size 2 --T_0 10 --epochs 80 --init_weights &

# for FHDMi
CUDA_VISIBLE_DEVICES="GPU_ID" nohup accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --dont_calc_mets_at_all --log2file \
    --data_path "path_to/FHDMi_complete" \
    --data_name fhdmi --train_dir "train" --test_dir "test" --moire_dir "source" --clean_dir "target" \
    --batch_size 2 --T_0 50 --epochs 200 --init_weights --num_branches 4 &

# for UHDM
CUDA_VISIBLE_DEVICES="GPU_ID" nohup accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --dont_calc_mets_at_all --log2file \
    --data_path "path_to/UHDM_DATA" \
    --data_name uhdm --train_dir "train" --test_dir "test" --moire_dir "" --clean_dir "" \
    --batch_size 2 --T_0 50 --epochs 200 --init_weights --num_branches 4 &
```

2. Finding the best checkpoints from the last training steps
```bash
# for LCDMoiré
for epoch in {190..199} ; do
  CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
        --data_path "path_to/aim2019_demoireing_track1" \
        --data_name aim --train_dir "train" --test_dir "val" --moire_dir "moire" --clean_dir "clear" \
        --resume "path_to/0${epoch}_checkpoint.pth.tar"
done

# for TIP2018
for epoch in {190..199} ; do
  CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
        --data_path "path_to/TIP2018_original" \
        --data_name tip18 --train_dir "trainData" --test_dir "testData" --moire_dir "source" --clean_dir "target" \
        --resume "path_to/0${epoch}_checkpoint.pth.tar"
done

# for FHDMi
for epoch in {190..199} ; do
  CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
        --data_path "path_to/FHDMi_complete" \
        --data_name fhdmi --train_dir "train" --test_dir "test" --moire_dir "source" --clean_dir "target" \
        --resume "path_to/0${epoch}_checkpoint.pth.tar" --num_branches 4
done

# for UHDM
for epoch in {190..199} ; do
  CUDA_VISIBLE_DEVICES="GPU_ID" accelerate launch --config_file default_config.yaml --mixed_precision=fp16 main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate --log2file \
          --data_path "path_to/UHDM_DATA" \
          --data_name uhdm --train_dir "train" --test_dir "test" --moire_dir "" --clean_dir "" \
          --resume "path_to/0${epoch}_checkpoint.pth.tar" --num_branches 4
done
```

# Citation
If you find this work useful for your research, please cite our paper:
```
@article{2023_nguyen_gad,
  author  = {Duong Hai, Nguyen and Se-Ho, Lee and Chul, Lee},
  title   = {Multiscale Guided Coarse-to-Fine Network for Screenshot Demoiréing},
  journal = {IEEE Signal Processing Letters},
  volume  = {},
  number  = {},
  pages   = {},
  month   = jul,
  year    = {2023},
  doi     = {10.1109/LSP.2023.3296039}
}
```

The code is released under the MIT license. See [LICENSE](https://choosealicense.com/licenses/mit/) for additional details.

# Acknowledgements
This code is built on [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/main/imagenet) and [UHDM](https://github.com/CVMI-Lab/UHDM).