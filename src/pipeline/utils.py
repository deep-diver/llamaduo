from datasets import DatasetDict
from huggingface_hub import HfApi
from ..utils.import_utils import is_alignment_available

if is_alignment_available():
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

def get_sha(model_id, revision):
    hf_api = HfApi()
    model_info = hf_api.model_info(
        model_id, revision=revision
    )
    return model_info.sha

def push_to_hf_hub(dataset_id, split, ds, append=True):
    """
    push_to_hf_hub pushes ds to the Hugging Face Dataset repository of
    dataset_id ID. If dataset_id does not exist, it creates one. If not, 
    and if append is set True, it appends ds to the existing one on the
    Dataset repository.
    """
    exist = False

    try:
        create_repo(dataset_id, repo_type="dataset")
    except HfHubHTTPError as e:
        exist = True
      
    if exist and append:
        existing_ds = load_dataset(dataset_id)
        concat_ds = concatenate_datasets([existing_ds[split], ds[split]])
        ds = DatasetDict({split: concat_ds})

    ds.push_to_hub(dataset_id)
