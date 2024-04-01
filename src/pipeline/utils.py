from alignment import (
    ModelArguments,
    DataArguments,
    H4ArgumentParser,
    SFTConfig
)

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import create_repo
from huggingface_hub.utils import HfHubHTTPError

def get_args(yaml_file_path):
    """
    get_args returns ModelArguments, DataArguments, SFTConfig from the
    configurations obtained after the model training
    """
    configs = H4ArgumentParser(
            (ModelArguments, DataArguments, SFTConfig)
        ).parse_yaml_file(yaml_file_path)
    
    return configs

def push_to_hf_hub(dataset_id, split, ds, hf_token, append=True):
    create_repo(dataset_id, repo_type="dataset", exist_ok=True, token=hf_token)
    
    if append:
        existing_ds = load_dataset(dataset_id)
        if split in existing_ds.keys():
            ds = concatenate_datasets([existing_ds[split], ds])

    ds.push_to_hub(dataset_id, token=hf_token)