# Bi-Directional Modality Fusion Network For Audio-Visual Event Localization

This repo holds the code for the work presented on ICASSP 2022 

# Prerequisites

We provide the implementation in PyTorch for the ease of use.

Install the requirements by runing the following command:

```ruby
pip install -r requirements.txt
```

# Code and Data Preparation

We highly appreciate YapengTian for the shared features and code.

## Download Features ##

Two kinds of features (i.e., Visual features and Audio features) are required for experiments.

- Visual Features: You can download the VGG visual features from here.
* Audio Features: You can download the VGG-like audio features from here.
+ Additional Features: You can download the features of background videos here, which are required for the experiments of the weakly-supervised setting.

After downloading the features, please place them into the ```data``` folder. The structure of the ```data```  folder is shown as follows:

```ruby
data
|——audio_features.h5
|——audio_feature_noisy.h5
|——labels.h5
|——labels_noisy.h5
|——mil_labels.h5
|——test_order.h5
|——train_order.h5
|——val_order.h5
|——visual_feature.h5
|——visual_feature_noisy.h5
```
## Download Datasets (Optional) ##

You can download the AVE dataset from the repo here.

# Training and testing BMFN in a fully-supervised setting 

# Training and testing BMFN in a Weakly-supervised setting

Similar to training the model in a fully-supervised setting, you can run training and testing using the following commands:

Training

```ruby
bash weak_train.sh
```

Evaluating

```ruby
bash weak_test.sh
```

# Citation

Please cite the following paper if you feel this repo useful to your research

```ruby
@inproceedings{inproceedings,
author = {Liu, Shuo and Quan, Weize and Liu, Yuan and Yan, Dong‐Ming},
year = {2022},
month = {03},
pages = {},
title = {Bi-Directional Modality Fusion Network For Audio-Visual Event Localization},
doi = {10.1109/ICASSP43922.2022.9746280}
}
```
