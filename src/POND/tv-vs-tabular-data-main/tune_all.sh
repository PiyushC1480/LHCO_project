#!/bin/bash -i

bash tune_model.sh catboost 0.8 openml
# bash tune_model.sh ftt 0.8
# bash tune_model.sh xgboost 0.8
# bash tune_model.sh tv7.4 0.8 binclass
bash tune_model.sh mlp 0.8 openml
# bash tune_model.sh ftt 0.8 openml
bash tune_model.sh xgboost 0.8 openml
bash tune_model.sh tv7.3 0.8 openml
bash tune_model.sh tv7.2 0.8 openml
# bash tune_model.sh tv7.4 0.8 openml
bash tune_model.sh dcnv2 0.8 openml
