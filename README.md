[![forthebadge](https://img.shields.io/badge/PyTorch-1.10.1-orange.svg)](https://forthebadge.com)

Winning Solution in [MIPI2022](https://mipi-challenge.org/#awards) Challenges on RGB+ToF Depth Completion

## Requirements  
Required 
* pytorch
* numpy
* pillow 
* opencv-python-headless 
* scipy    
* Matplotlib
* torch_ema

Optional
* tqdm 
* tensorboardX

## Pre-trained models
Download the pretrained models from [Google Drive](https://drive.google.com/file/d/1Zby9acKFEbFzcC5lieECOlMN3yiMuhUm/view?usp=sharing)

## Quickstart

### Training

1. **Step 1:** download [training data](https://drive.google.com/file/d/1OkuUhlv5i5EIh5y7bgYTt_5ZRGF__1aT/view) and fixed [validation data](https://drive.google.com/file/d/1ki4oIJmY-AKPSg_T1214kb0Z_2g7ma8-/view?usp=sharing) from Google Drive and unzip them.

2. **Step 2:**

   - **Train set:** Record the path of the data pairs to a text file like [this](https://github.com/dwHou/EMDC-PyTorch/blob/main/data_train.list) and assign the file location to the variable <font color="brown">*'train_txt'*</font>  in  <font color="brown">*./utils/dataset.py*</font>. Also, modify the data directory path in the member function <font color="brown">*'self._load_png'*</font>.
   - **Val set:** Processing is similar to the above.
   - **Note that** <font color="brown">*'BeachApartmentInterior_My_ir'*</font> scene's folder is removed from the training set, as it is partitioned into the fixed validation set.

3. **Step 3:**

   ```shell
   bash train.sh
   ```

### Test

1. **Step1:**

   download the official [test data](https://drive.google.com/file/d/13hdstLpsp8uMfkwYYCmlJwKd-a4Ed719/view) and put it in <font color="blue">*./Submit*</font>

   download the pretrained model and put it in <font color="blue">*./checkpoints*</font>

2. **Step2:**

   ```shell
   cd ./Submit
   cp ../utils/define_model.py ./
   cp -R ../models ./
   bash test.sh 
   ```

3. **Step 3:** Check the results under the path <font color="brown">*./Submit/results*</font>

## Citation
If you find our codes useful for your research, please consider citing our paper:
(TBD)

[1] Dewang Hou, Yuanyuan Du, Kai Zhao, and Yang Zhao, **"Learning an Efficient Multimodal Depth Completion Model",** <i>1st MIPI: Mobile Intelligent Photography & Imaging workshop and challenge on RGB+ToF depth completion in conjunction with **ECCV 2022**. </i> [[PDF]()] [[arXiv](https://arxiv.org/abs/2208.10771)]

```bibtex
@article{xx,
    author = {xx},
    title = {xx},
    year = {2022}
}
```

