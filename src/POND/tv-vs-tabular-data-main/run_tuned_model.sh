#!/bin/bash -i

# binclass downstream datasets
# DATASETS=("Bank_Personal_Loan_Modelling" "Churn_Modelling")
# skipped dataset because of cuda out of memory error
# DATASETS=('train_1142_Sick_numeric' 'trial' 'train_1451_early-stage-diabetes' 'train_1752_Wisconsin-breast-cancer-cytology-features' 'diabetes_data_upload' 'train_1635_Is-this-a-good-customer' 'train_1619_NBA-2k20-player-dataset' 'loan_train' 'train_1461_heart-failure' 'train_2703_compas-two-years' 'train_1006_Titanic' 'train_1408_national-longitudinal-survey-binary' 'TravelInsurancePrediction' 'Employee Satisfaction Index' 'train_0472_analcatdata_marketing' 'train_0408_pharynx' 'train_1774_Early-Stage-Diabetes-Risk-Prediction-Dataset' 'train_0400_analcatdata_supreme' 'train_1512_eye_movements' 'train_1898_Personal-Loan-Modeling' 'b_depressed' 'train_1692_Gender-Classification-Dataset' 'train_1011_cleve' 'train_0885_compas-two-years' 'train_1564_Mammographic-Mass-Data-Set' 'UniversalBank' 'Breast_Cancer' 'train_0356_delta_elevators' 'Bank_Personal_Loan_Modelling' 'train_1742_Loan-Predication' 'train_0124_analcatdata_impeach')

model=$1
model_="\<${1}\>"
reduce_data=$2
task=$3 # either binclass | openml

data_dir=./data/finetune-bin
DNN=("mlp" "tv6" "tv6.1" "tv6.2" "tv6.3" "tv6.4" "tv7" "tv7.1" "tv7.2" "tv7.3" "tv7.4" "autoint" "dcnv2" "saint")
TREE=("xgboost" "catboost" "tabnet")
delim='*/'
suffix='.csv'
echo "For model: $model, reduce_data: $reduce_data"

if [[ "$task" == "binclass" ]]; then
	iter="$data_dir"/*
else
	IFS=$'\n'
	iter=$(cat "openml_datasets_sorted.txt")
	# iter=$(cat "openml_X_num_none.txt")
	# iter=$(cat "openml_ftt_miss.txt")
fi

for DATASET in $iter; do
	# for DATASET in "${DATASETS[@]}"; do
	if [[ "$task" == "binclass" ]]; then

		file_name="${DATASET/#${delim}/}"

		# Skipping non csv files (like cache)
		if [ "${file_name: -4}" != "$suffix" ]; then
			continue
		fi
		file_name=${file_name%"${suffix}"}
	else
		file_name="$DATASET"
	fi
	echo "For dataste: $file_name"
	# non-LM DNNs (MLP, AutoInt, DCN2, SAINT)
	if [[ ${DNN[@]} =~ $model_ ]]; then
		python scripts/finetune/tuned/run_tuned_config_dnns.py \
			--model "$model" \
			--dataset "$file_name" \
			--task "$task" \
			--reduce_data "$reduce_data"
	fi
	if [[ ${TREE[@]} =~ $model_ ]]; then
		python scripts/finetune/tuned/run_tuned_config_tree.py \
			--model "$model" \
			--dataset "$file_name" \
			--task "$task" \
			--reduce_data "$reduce_data"
	fi
	if [[ "$model" == "ftt" ]]; then
		python scripts/finetune/tuned/run_tuned_config_ftt.py \
			--dataset "$file_name" \
			--task "$task" \
			--reduce_data "$reduce_data"
	fi
done
