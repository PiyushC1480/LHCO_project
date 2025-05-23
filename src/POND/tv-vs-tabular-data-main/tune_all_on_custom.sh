#!/bin/bash -i
# Option to add custom datasets and run all the models on them.
# For example, an mnist dataset is added and task is set to custom.
#

# python scripts/finetune/tune/tune_dnns.py --model tv7.3 --dataset mnist --task custom --reduce_data 0.8
# python scripts/finetune/tune/tune_dnns.py --model dcnv2 --dataset mnist --task custom --reduce_data 0.8
# python scripts/finetune/tune/tune_dnns.py --model mlp --dataset mnist --task custom --reduce_data 0.8

# python scripts/finetune/tune/tune_ftt.py --dataset mnist --task custom --reduce_data 0.8
python scripts/finetune/tune/tune_trees.py --model catboost --dataset mnist --task custom --reduce_data 0.8
python scripts/finetune/tune/tune_trees.py --model xgboost --dataset mnist --task custom --reduce_data 0.8
