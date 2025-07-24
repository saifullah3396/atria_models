#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCSETS_DIR=$SCRIPT_DIR/../../
ATRIA_DIR=$DOCSETS_DIR/../atria
ATRIA_CORE_DIR=$DOCSETS_DIR/../atria_core
PYTHONPATH=$DOCSETS_DIR/src:$ATRIA_DIR/src:$ATRIA_CORE_DIR/src:$PYTHONPATH
DATASET_NAME=$1

declare -a available_datasets=(
    "classification tobacco3482"
    "classification rvlcdip"
)

declare -a checkpoints=(
    "/media/aletheia/a865e985-032a-4793-9899-2063093eac27/home/ataraxia/models/models/document_cls/layoutlmv3/tobacco3482.pt"
    "/media/aletheia/a865e985-032a-4793-9899-2063093eac27/home/ataraxia/models/models/document_cls/layoutlmv3/rvlcdip.pt"
)

train() {
    local dataset_name="$1"
    local additional_args="${@:2}"

    for i in "${!available_datasets[@]}"; do
        IFS=' ' read -r task dataset_with_numbers config_name <<<"${available_datasets[$i]}"
        checkpoint="${checkpoints[$i]}"

        # check if config_name is empty
        if [[ -z "$config_name" ]]; then
            dataset_path=$task/$dataset_name
        else
            dataset_path=$task/$dataset_name/$config_name            
        fi

        if [[ "$dataset_with_numbers" == "$dataset_name" ]]; then
            set -x
            PYTHONPATH=$PYTHONPATH python -m atria.task_pipelines.trainer \
                --config-name task_pipeline/trainer/sequence_classification \
                hydra.searchpath=[pkg://docsets/conf] \
                dataset@data_pipeline.dataset=$dataset_path \
                data_pipeline.data_dir=$DEFAULT_ATRIA_CACHE_DIR/datasets/$task/$dataset_name/ \
                model@model_pipeline.model=layoutlmv3_for_sequence_classification \
                model_pipeline.checkpoint_configs='[{'checkpoint_path': '$checkpoint', 'checkpoint_state_dict_path': 'model' ,'model_state_dict_path': '_model'}]' \
                data_pipeline.train_dataloader.batch_size=8 \
                data_pipeline.evaluation_dataloader.batch_size=16 \
                data_pipeline.runtime_transforms.train.tokenizer_name=microsoft/layoutlmv3-base \
                data_pipeline.runtime_transforms.evaluation.tokenizer_name=microsoft/layoutlmv3-base \
                do_train=False \
                $additional_args
            set +x
            return 0
        fi
    done

    echo "Dataset '$dataset_name' not found."
    return 1
}

if [ -z "$DATASET_NAME" ]; then
    echo "No dataset name provided. Please provide a dataset name or use 'all'."
    exit 1
elif [[ "$DATASET_NAME" == "all" ]]; then
    echo "Running train on all datasets..."
    for i in "${!available_datasets[@]}"; do
        IFS=' ' read -r _ dataset_with_numbers _ <<<"${available_datasets[$i]}"
        train "$dataset_with_numbers" "${@:2}"
    done
    exit 0
else
    train "$DATASET_NAME" "${@:2}"
fi
