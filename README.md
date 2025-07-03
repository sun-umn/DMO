# Official implementation of Exact Reformulation and Optimization for Binary Imbalanced Classification

## Quick Start

### Installation
___
#### Pull Git Repo
```bash
git clone git@github.com:sun-umn/DMO.git
```

#### Prepare Environment

```bash
conda env update -n dmo --file env.yml
conda activate dmo
```

### Prepare Datasets

#### Download Dataset

| Dataset Name        | Download Link                                                                                   |
|---------------------|------------------------------------------------------------------------------------------------|
| UCI                 | [Download](https://drive.google.com/drive/folders/1YFGuq_9QFSIUysHLWzJX7Et_VXQXPkfh?usp=sharing) |
| Fire                | [Download](https://www.kaggle.com/datasets/phylake1337/fire-dataset/data)                      |
| Eyepacs             | [Download](https://www.kaggle.com/c/diabetic-retinopathy-detection/)                           |
| ADE-corpus-V2       | [Download](https://huggingface.co/datasets/ade_corpus_v2)                                      |



```bash
cd DMO/
mkdir datasets/
mv [dataset] datasets/ # move downloaded dataset to DMO/datasets/
```


### Examples

Train MLP model
```bash
# Fix precision at real, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python FPOR.py --ds wilt --alpha 0.8 --seed 0

# Fix recall at precision, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python FROP.py --ds wilt --alpha 0.8 --seed 0

# Optimize F-beta score, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python OFBS.py --ds wilt --seed 0
```


Train linear model
```bash
# Fix precision at real, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python FPOR.py --ds eyepacs --alpha 0.8 --linear --seed 0 

# Fix recall at precision, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python FROP.py --ds eyepacs --alpha 0.8 --linear --seed 0

# Optimize F-beta score, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python OFBS.py --ds eyepacs --linear --seed 0
```

Example of run log (FROP for Eyepacs)

```bash
=========== iter 49 =============
metric constraint: 4.89354133605957e-05
ind_constr: max 5.848182354384335e-06    mean 1.975642316409676e-09
max mu: 119253.234375    mean mu: 3.8800952434539795
confidence: 0.009909236803650856         lam: 248964.6114895642
proxy_precision: 0.45377567410469055     proxy_recall: 0.7999510765075684        proxy fb_score: 0.5790709257125854
precision: 0.45275014638900757   recall: 0.7966474890708923      fb_score: 0.5773698687553406
--------------------- validation set --------------------
precision: 0.4125060737133026    recall: 0.7445319294929504      fb_score: 0.5308796167373657
val obj: -0.4125060737133026     constr: 0.055468082427978516    best obj: -0.41230618953704834          best feasible obj: 0



train Final:
precision: 0.4517207741737366    recall: 0.7933439612388611      f1: 0.575664758682251
val final:
precision: 0.41230618953704834   recall: 0.7445319294929504      f1: 0.5307140350341797
test final:
precision: 0.4286929965019226    recall: 0.7175105214118958      f1: 0.5367139577865601

```

## How to cite this work
___

If you find this gitrepo useful, please consider citing the associated paper using the snippet below:
```bibtex
@inproceedings{travadi2023direct,
  title={Direct Metric Optimization for Imbalanced Classification},
  author={Travadi, Yash and Peng, Le and Cui, Ying and Sun, Ju},
  booktitle={2023 IEEE 11th International Conference on Healthcare Informatics (ICHI)},
  pages={698--700},
  year={2023},
  organization={IEEE}
}
```
