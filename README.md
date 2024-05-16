<h1 align="center">
    🛣️ MULTI-LANE 🛣️
</h1>
<p align="center">
    <img src="assets/logo.png" alt="drawing" style="width:200px;"/>
</p>

Official Implementation of "Less is more: Summarizing Patch Tokens for efficient Multi-Label Class-Incremental Learning (MULTI-LANE)", published at 3rd Conference on Lifelong Learning Agents (CoLLAs 2024)

## 🛠️ Installation
Create and create a conda environment with Python 3.8.17:
```
$ conda create -n multilane python=3.8.17
$ conda activate multilane
```

Install Python requirements from file:
```
$ pip install -r requirements.txt
```

## 🏋️ Datasets and Pre-Trained Weights
Both are automatically downloaded at training time. 

Datasets are stored in the `datasets/` folder (it will be automatically created). To specify a different folder, pass the argument `--data_path /path/to/datasets` when launching the training script.

Pre-Trained Weights are stored in the default timm directory.

## 🏃‍♀️ Training
We prepared six training scripts to train MULTI-LANE on each dataset and configuration we show in the paper:

- COCO B0-C10 (MLCIL) `$ ./train_coco.sh`
- COCO B40-C10 (MLCIL) `$ ./train_coco_40.sh`
- VOC B0-C4 (MLCIL) `$ ./train_voc.sh`
- VOC B10-C2 (MLCIL) `$ ./train_voc_10.sh`
- ImageNet-R (CIL) `$ ./train_inr.sh`
- CIFAR-100 (CIL) `$ ./train_c100.sh`

## 🗿 Changing the number of Selectors
To change the number of selectors at training time, pass the argument `--num_selectors` with the chosen amount of selectors. For example, to train on coco with 1 selector, run:
```
$ ./train_coco.sh --num_selectors 1
```
Or edit train_coco.sh and add the argument manually.

Similarly, to use token merging instead of selectors pass the argument `--tome` with the number of tokens to be retained. Additionally, remove selectors from training by setting their quantity to 0 `--num_selectors 0`. For example:
```
$ ./train_coco.sh --num_selectors 0 --tome 24
```

## 🙅‍♀️Ablations
To run one of the ablations we show in the paper, use the following arguments:
- Disabling Normalization `$ ./train_coco.sh --normalize 'none'`
- Disabling Parallel Pathways (use query-key selection mechanism) `$ ./train_coco.sh --head_model 'task'`
- Disabling Drop and Replace `$ ./train_coco.sh --disable_dandr`

## Citing our work
```
todo
```
