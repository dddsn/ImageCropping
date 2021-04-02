# image_cropping

This is an implementation of Weakly Supervised Real-time Image Cropping based on Aesthetic Distributions (MM 2020).

## 1. Pre-request

### 1.1. Environment
```bash
pip install -r requirement.txt
```

### 1.2. Datasets
FCD
FLMS
CUHK-ICD

## 2. Test

### 2.1. Evaluation
```bash
bash run.sh
```
or
```bash
python eval.py --dataset_path you-datasets-path --weight weights/gan_weights.h5 --log eval_gan.txt
```

### 2.2. Demo
```bash
python demo.py --weight weights/gan_weights.h5 --img_path images --save_path result --log 1
```

## 3. License & Citation

You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing the following paper** and **indicating any changes** that you've made.

```tex
@inproceedings{lu2020weakly,
  title={Weakly Supervised Real-time Image Cropping based on Aesthetic Distributions},
  author={Lu, Peng and Liu, Jiahui and Peng, Xujun and Wang, Xiaojie},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={120--128},
  year={2020}
}
```
