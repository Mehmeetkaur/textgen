#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

declare -a small_datasets=(
    "cifar10/1k" # tested done
    # "huggingface_cifar10/plain_text_1k" # tested done
    # "mnist/mnist_1k" # tested done
    # "tobacco3482/image_with_ocr" # tested done
    # "rvlcdip/image_with_ocr_1k" # tested done
    # "cord/default" # tested done
    # "funsd/default" # tested done
    # "sroie/default" # tested done
    # "wild_receipts/default" # tested done
    # "docile/kile" # tested done
    # "fintabnet/1k"  # tested done
    # "icdar2019/trackA_modern" # tested done
    # "icdar2013/default" # tested done
    # "docvqa/default" 
    # "docbank/0.1k"  # too big, failing downloads
)


declare -a big_datasets=(
    "rvlcdip/image_with_ocr" 
    "docbank/default"  
    "doclaynet/2022.08" # too big, failing downloads, untested
    "publaynet/default" # too big, failing downloads, untested
    "pubtables1m/detection_1k" # too big, failing downloads, untested
    "pubtables1m/structure_1k" # too big, failing downloads, untested
)


if [[ "$1" == "small_datasets" ]]; then
    for dataset_entry in "${small_datasets[@]}"; do
        echo "Processing dataset: $name with config: $config and args: ${@:2}"
        uv run python -m atria_datasets.prepare_dataset $dataset_entry ${@:2} --upload_to_hub=True
    done
elif [[ "$1" == "big_datasets" ]]; then
    for dataset_entry in "${big_datasets[@]}"; do
        echo "Processing dataset: $name with config: $config and args: ${@:2}"
        uv run python -m atria_datasets.prepare_dataset $dataset_entry ${@:2} --upload_to_hub=True
    done
else
    dataset_entry="$1"
    echo "Processing dataset: $name with config: $config and args: ${@:2}"
    uv run python -m atria_datasets.prepare_dataset $dataset_entry ${@:2} --upload_to_hub=True
fi