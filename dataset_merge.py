import os
import argparse
from datasets import (
    DatasetDict,
    load_dataset, 
    concatenate_datasets
)

from huggingface_hub import create_repo
from huggingface_hub.utils import HfHubHTTPError
from utils import update_args

KEEP_COLUMNS = ["generator", "prompt_id", "prompt", "messages", "category"]

def check_args(args):
    if args.first_ds_id is None or \
        args.first_ds_train_split is None or \
        args.first_ds_test_split is None:
        raise ValueError("some arguments for the first dataset are missing")

    if args.second_ds_id is None or \
        args.second_ds_train_split is None:
        raise ValueError("some arguments for the second dataset are missing")

    if args.result_ds_id is None or \
        args.result_ds_train_split is None or \
        args.result_ds_test_split is None:
        raise ValueError("some arguments for the result dataset are missing")

def merge_datasets(args):
    try:
        check_args(args)
    except ValueError as e:
        print(str(e))

    first_train_ds = load_dataset(args.first_ds_id, split=args.first_ds_train_split)
    first_test_ds = load_dataset(args.first_ds_id, split=args.first_ds_test_split)

    second_train_ds = load_dataset(args.second_ds_id, split=args.second_ds_train_split)
    result_train_ds = concatenate_datasets([first_train_ds, second_train_ds])
    result_test_ds = first_test_ds

    if args.second_ds_test_split:
        second_test_ds = load_dataset(args.second_ds_id, split=args.second_ds_test_split)
        result_test_ds = concatenate_datasets([first_test_ds, second_test_ds])

    result_ds = DatasetDict(
        {
            args.result_ds_train_split: result_train_ds,
            args.result_ds_test_split: result_test_ds
        }
    )

    if args.push_result_ds_to_hf_hub and args.hf_token is not None:
        exist = False

        try:
            create_repo(args.result_ds_id, repo_type="dataset", token=args.hf_token)
        except HfHubHTTPError as e:
            exist = True

        if exist:
            if args.result_ds_train_append:
                result_ds[args.result_ds_train_split] = concatenate_datasets(
                    [
                        result_ds[args.result_ds_train_split], 
                        load_dataset(args.result_ds_id, split=args.result_ds_train_split)
                    ]
                )

            if args.result_ds_test_append:
                result_ds[args.result_ds_test_split] = concatenate_datasets(
                    [
                        result_ds[args.result_ds_test_split], 
                        load_dataset(args.result_ds_id, split=args.result_ds_test_split)
                    ]
                )

        result_ds.push_to_hub(
            args.result_ds_id, token=args.hf_token
        )

    else:
        result_ds.save_to_disk(args.result_ds_id)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="CLI for merging two datasets")

    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--from-config", type=str, default="config/dataset_merge.yaml",
                        help="set CLI options from YAML config")

    parser.add_argument("--first-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID of the first dataset")
    parser.add_argument("--first-ds-train-split", type=str, default=None,
                        help="train split to merge from the first dataset")
    parser.add_argument("--first-ds-test-split", type=str, default=None,
                        help="test split to merge from the first dataset")

    parser.add_argument("--second-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID of the second dataset")
    parser.add_argument("--second-ds-train-split", type=str, default=None,
                        help="train split to merge from the second dataset")
    parser.add_argument("--second-ds-test-split", type=str, default=None,
                        help="test split to merge from the second dataset")

    parser.add_argument("--push-result-ds-to-hf-hub", action="store_true",
                        help="Whether to push result dataset to Hugging Face Dataset repository(Hub)")
    parser.add_argument("--result-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID of the result dataset")
    parser.add_argument("--result-ds-train-split", type=str, default=None,
                        help="train split of the resulting dataset")
    parser.add_argument("--result-ds-test-split", type=str, default=None,
                        help="test split of the resulting dataset")
    parser.add_argument("--result-ds-train-append", action="store_true", default=True,
                        help="Wheter to overwrite or append on the resulting dataset on train split")
    parser.add_argument("--result-ds-test-append", action="store_true", default=True,
                        help="Wheter to overwrite or append on the resulting dataset on test split")             

    args = parser.parse_args()
    args = update_args(parser, args)

    merge_datasets(args)
