# PTAMSC: Pure Transformer-flavored Augmented Multi-modal Sentiment Classifier

Author: GONGGONGJOHN

This is the official repository for the final project of DaSE undergraduate course "Contemporary Artificial Intelligence": Multimodal Sentiment Analysis. All the scripts and results are supposed to be executable and repoducible.

## Setup

This project is implemented by Python 3.8, all related dependencies have been exported to `requirements.txt`.
Use the following command to install all the required packages automatically:
```shell
pip -r requirements.txt
```

The main dependencies includes:

- chardet 4.0.0
- matplotlib 3.3.2
- Pillow 9.2.0
- scikit_learn 1.1.1
- torch 1.9.0
- torchvision 0.10.0
- tqdm 4.64.0
- transformers 4.20.1

## Repository Structure

```
.
├── README.md
├── assets
│   └── ptamsc_structure.png
├── baseline.py
├── baseline_model.py
├── bert_baseline.py
├── bert_train.py
├── data_pipe.py
├── data_utils.py
├── main.py
├── model.py
├── plot
│   ├── fuse_accuracy.png
│   ├── plot_clmlf_pretrain.py
│   ├── plot_clmlf_train.py
│   ├── plot_fuse_accuracy.py
│   ├── plot_hypertune.py
│   ├── plot_resnet_f1.py
├── resnet_train.py
├── swin_baseline.py
├── swin_train.py
├── xlm_baseline.py
├── xlm_hypertune.py
└── xlm_train.py
```

## Model Overview

PTAMSC is a pure transformer-based image-text multimodal classification model. The main structure is as follows:

![structure](assets/ptamsc_structure.png)

## Performance

| Model            | Accuracy  | Precision | Recall    | F1        |
|------------------|-----------|-----------|-----------|-----------|
| Bert             | 72.99     | 65.56     | 54.47     | 55.66     |
| XLMRoBERTa       | 73.58     | 67.62     | 59.48     | 61.52     |
| ResNet           | 57.93     | 36.01     | 33.71     | 25.59     |
| Swin Transformer | 66.73     | 57.90     | 54.02     | 55.38     |
| X+S(Concatenate) | 70.25     | 63.74     | 54.88     | 57.12     |
| X+S(Additive)    | 71.23     | 66.31     | 54.79     | 55.01     |
| **PTAMSC**       | **76.51** | **70.88** | **63.34** | **65.69** |


## Reproduce Experiment

### Dataset Preparation

To reproduce experiments on the original dataset, please use the following structure to organize your dataset directory:

```
.
├── [Dataset root]
│   ├── train.txt  # Training label
│   ├── test_without_label.txt  # Test items to be predicted
│   ├── source
│   │   ├── x.txt  # Paired text-image data goes here
│   │   ├── x.jpg
└── └── └── xlm_train.py
```

As illustrated in the project report, the original dataset contains text instances not encoded in UTF-8/ANSI, 
please open them up with GBK encoding and translate them into English manually.
Otherwise there will be errors when reading data instances into the model.

### Baseline Model

```shell
python baseline.py --model [Name of the baseline model] --data_path [Dataset Root Directory]
```

The optional parameters and its meanings are as follows:

- \--model: The name of the baseline model to be used for training and validation, can be one of: bert, xlmroberta, resnet, swin, concat, additive
- \--img_scale_size: The target size of images to feed into the model, default: 224
- \--text_model_name: The pretrained model name of the text embedding model, default: xlm-roberta-base
- \--batch_size: The batch size of all the datasets, default: 16


### Main Model

```shell
python main.py --do_train --do_eval --do_test --data_path [Dataset Root Directory]
```

The optional parameters and its meanings are as follows:

- \--img_scale_size: The target size of images to feed into the model, default: 224
- \--text_model_name: The pretrained model name of the text embedding model, default: xlm-roberta-base
- \--batch_size: The batch size of all the datasets, default: 16

## References

We only list some major reference papers here, the full references list can be found in the project report:

1. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2015.
2. Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451, Online, July 2020. Association for Computational Linguistics.
3. Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
4. Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. CLMLF:a contrastive learning and multi-layer fusion method for multimodal sentiment detection. In Findings of the Association for Computational Linguistics: NAACL 2022. Association for Computational Linguistics, 2022.
5. Yan Ling, Jianfei Yu, and Rui Xia. Vision-language pre-training for multimodal aspect-based sentiment analysis. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2149– 2159, Dublin, Ireland, May 2022. Association for Computational Linguistics.
6. Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. CoRR, abs/1807.03748, 2018.
7. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

