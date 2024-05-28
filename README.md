# Official implementation of Exact Reformulation and Optimization for Binary Imbalanced Classification

## Quick Start

### Installation
___
#### Pull Git Repo
```bash
git clone git@github.com:PL97/DMO.git
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
| UCI                 | [Download](https://drive.google.com/drive/folders/1NBHcQohoCJg7gKkvGRC_BCYdCUEwYkq-?usp=drive_link) |
| Fire                | [Download](https://www.kaggle.com/datasets/phylake1337/fire-dataset/data)                      |
| Eyepacs             | [Download](https://www.kaggle.com/c/diabetic-retinopathy-detection/)                           |
| ADE-corpus-V2       | [Download](https://huggingface.co/datasets/ade_corpus_v2)                                      |


```bash
mkdir data/
mv [dataset] data/
```


### Examples

```bash
# Fix precision at real, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python FPOR.py --ds wilt --alpha 0.8 --seed 0

# Fix recall at precision, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python FROP.py --ds wilt --alpha 0.8 --seed 0

# Optimize F-beta score, using wilt dataset, with a prefix threshold 0f 0.8, using a random seed 0
python OFBS.py --ds wilt --seed 0
```

```bash
=========99/100===============
lambda: 3.226607916197264e+23, 4.729280783717073e+25, [3.226608e+23]
violation: 3.321038093417883e-05, 3.321038093417883e-05, [0.0181669]
real obj: [[0.5915493]]                  const: [[0.75]]
estimated obj: [[0.616051]]              const: [[0.7818332]]

=========================final evaluation===============================
Train: real obj: [0.61971831]            const: [0.76300578]
Test: real obj: [0.70833333]             const: [0.77272727]

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
