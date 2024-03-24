from alignment import (
    ModelArguments,
    DataArguments,
    H4ArgumentParser,
    SFTConfig
)

def get_args(yaml_file_path):
    configs = H4ArgumentParser(
            (ModelArguments, DataArguments, SFTConfig)
        ).parse_yaml_file(yaml_file_path)
    
    return configs