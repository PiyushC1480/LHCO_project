This repo is cloned from repository of the paper TP-BERTa [Making Pre-trained Language Models Great on Tabular Prediction](https://openreview.net/pdf?id=anzIzGZuLi) (ICLR 2024 Spotlight)


# This is the official repository of the algorithm POND
This work was done by Harsh Pandey, IITD MS by Research, under the guidance of Prof. Amitabha Bagchi and Prof. Srikanta Bedathur.



## Project Structure

The repo structure and module functions are as follows:

```
├─bin ---- // Implementation of tabular models
│ ├─tv7.py ---- // A version of POND (earlier named TV) base class
│ ├─tv6.py ---- // A version of POND (earlier named TV) base class
│ ├─tv7_shared.py ---- // An independent class of TV/POND with sampling of kernel points and others biult-in.
│ ├─tpberta_modeling.py ---- // TP-BERTa base class
│ └─xxx.py ----------------- // Other non-LM DNN baselines
├─lib ---- // Utilities (Same as that of TP-BERTa)
│ ├─aux.py --------------- // Auxiliary Loss: Magnitude-aware Triplet Loss
│ ├─feature_encoder.py --- // Numerical Value Binner (C4.5 discretization)
│ ├─optim.py ------------- // Utilities for optimizer & trainer
│ ├─env.py --------------- // Environment Variables configs
│ ├─data.py -------------- // Dataset & Data Transformation class
│ ├─data_utils.py -------- // Data Config & Multi-task Loader class
│ └─xxx.py --------------- // Other standard utils
├─data --- // csv file path for pre-training & fine-tuning
│ ├─pretrain-bin
│ ├─pretrain-reg
│ ├─finetune-bin
│ ├─finetune-reg
│ └─finetune-mul
├─checkpoints --- // Configs for dataset and unused configs of Pre-trained model weights & configs (RoBERTa, TP-BERTa)
├─configs --- // Model & Training configs for TV/POND and baselines
│ ├─default --- // default configs
│ └─tuned ----- // tuned configs (generated with hyperparameter tuning scripts)
├─scripts --- // Experiment codes
│ ├─pretrain --- // Codes for TP-BERTa pre-training
│ ├─finetune --- // Codes for baseline fine-tuning & hyperparameter tuning
│ ├─parameter_effect --- // Codes for checking the effect of parameters on models.
│ ├─clean_feat_names.py --- // Deprecated, Text clean for table feature names
│ └─examples --- // Deprecated, For examples to run the scripts look `run_tuned_all.sh` and 'tune_all.sh'
├─tune_model.sh --- // To tune any model on different hyperparameter for a given dataset with reduce data parameter which reduces the given fraction of dataset.
├─tune_all.sh --- // To tune all models in one go
├─run_tuned_model.sh --- // To run a tuned model (on multiple hyperparameter) on a given dataset with reduce data parameter which reduce the given fraction of dataset.
├─run_tuned_all.sh --- // To tune all models in one go
├─param_effect.sh --- // To test the effect of changing the parameters of model, especially to test how the size of parameter affects model's performance.
└─inference --- // Scripts to generate plots and perform inference
  ├─finetuned_poots --- // This folder has dataset wise final results of the models performance
  └─combine_tuned_results.ipynb --- // This file has 
```

## Dependencies

All necessary dependencies of TP-BERTa are included in `requirement.txt`, there might be few more basic dependencies which should be installed if the execution fails.


## Datasets
We use 2 sets of datasets, one is a subset from OpenML-CC18 datasets for binary classification and the other is the dataset used in the paper TP-BERTa.

### OpenML Dataset
- No need to download this dataset, the files `/lib/data_openml.py` and `/lib/data_utils_openml.py` automatically handles it.

### TabPertNet Dataset
- We used the dataset used in TP-BERTa paper, which is a subset from [TabPertNet (OpenTabs currently)](https://arxiv.org/abs/2307.04308) datasets.
- Download datasets for [pre-training](https://drive.google.com/uc?export=download&id=1Jy45I_vTKn6McMROi5IKjKoSi9QJtx9A) (202 datasets) and [fine-tuning](https://drive.google.com/uc?export=download&id=1JhOJR1kxjyu4w4ZHi8VcxgMh-iYJRDgG) (145 datasets).
- Unzip the `*.tar.gz` file to the `data` folder (create one if not exists).


## Acknowledgments

Our codes are influenced by the following repos:

- [TP-BERTa](https://github.com/jyansir/tp-berta)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [RTDL Numerical Embeddings](https://github.com/yandex-research/rtdl-num-embeddings)
