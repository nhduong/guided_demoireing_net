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
    <br>
    <a href="https://nhduong.github.io/guided_demoireing_net">Project Page</a>
    ·
    <a href="...">Technical Report</a>
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
1. Download pretrained models

| Dataset | Pretrained Model |
| :---: | :---: |
| LCDMoiré | [Google Drive](https://drive.google.com/drive/folders/???) |
| TIP2018 | [Google Drive](https://drive.google.com/drive/folders/???) |
| FHDMi | [Google Drive](https://drive.google.com/drive/folders/???) |
| UHDM | [Google Drive](https://drive.google.com/drive/folders/???) |

2. Execute the following commands:
```bash
# for LCDMoiré
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate \
    --data_path "path-to/aim2019_demoireing_track1" \
    --data_name aim --train_dir "train" --test_dir "val" --moire_dir "moire" --clean_dir "clear" \
    --resume "path-to/pretrained_models/aim" --evaluate_epochs 0

# for TIP2018
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate \
    --data_path "path-to/TIP2018_original" \
    --data_name tip18 --train_dir "trainData" --test_dir "testData" --moire_dir "source" --clean_dir "target" \
    --resume "path-to/pretrained_models/tip18" --evaluate_epochs 0

# for FHDMi
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate \
    --data_path "path-to/FHDMi_complete" \
    --data_name fhdmi --train_dir "train" --test_dir "test" --moire_dir "source" --clean_dir "target" \
    --resume "path-to/pretrained_models/fhdmi" --evaluate_epochs 0 --num_branches 4

# for UHDM
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --test_batch_size 1 --affine --l1loss --adaloss --perloss --evaluate \
    --data_path "path-to/UHDM_DATA" \
    --data_name uhdm --train_dir "train" --test_dir "test" --moire_dir "" --clean_dir "" \
    --resume "path-to/pretrained_models/uhdm" --evaluate_epochs 0 --num_branches 4

```

# Training

Run the following commands:

```bash
# for LCDMoiré
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --batch_size 2 --workers 4 --exp_name SPL --data_name aim --T_0 50 --print_freq 1000 --train_dir "train" --test_dir "val" --moire_dir "moire" --clean_dir "clear" --dont_calc_mets_at_all --epochs 200 --test_batch_size 1 --note testing --affine --l1loss --adaloss --perloss --data_path "path-to/aim2019_demoireing_track1"

# for TIP2018
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --batch_size 2 --workers 4 --exp_name SPL --data_name tip18 --T_0 10 --print_freq 1000 --train_dir "trainData" --test_dir "testData" --moire_dir "source" --clean_dir "target" --dont_calc_mets_at_all --epochs 80 --test_batch_size 1 --note testing --affine --l1loss --adaloss --perloss --data_path "path-to/TIP2018_original"

# for FHDMi
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --batch_size 2 --workers 4 --exp_name SPL --data_name fhdmi --T_0 50 --print_freq 1000 --train_dir "train" --test_dir "test" --moire_dir "source" --clean_dir "target" --dont_calc_mets_at_all --epochs 200 --test_batch_size 1 --note testing --affine --l1loss --adaloss --perloss --data_path "path-to/FHDMi_complete"

# for UHDM
CUDA_VISIBLE_DEVICES="GPU_ID" python main.py --batch_size 2 --workers 4 --exp_name SPL --data_name uhdm --T_0 50 --print_freq 1000 --train_dir "train" --test_dir "test" --moire_dir "" --clean_dir "" --dont_calc_mets_at_all --epochs 200 --test_batch_size 1 --note testing --affine --l1loss --adaloss --perloss --data_path "path-to/UHDM_DATA"
```

# Citation
If you find this work useful for your research, please cite our paper:
```
@article{2023_nguyen_gad,
  title={Multiscale Guided Coarse-to-Fine Network for Screenshot Demoiréing},
  author={Nguyen, Duong Hai and Lee, Se-Ho and Lee, Chul},
  journal={submitted},
  year={2023}
}
```

The code is released under the MIT license. See [LICENSE](https://choosealicense.com/licenses/mit/) for additional details.

# Acknowledgements
This code is built on [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/main/imagenet) and [UHDM](https://github.com/CVMI-Lab/UHDM).