import yaml

def update_args(parser, args):
    """
    update_args replaces default values of the args. Arguments that are directly
    specified from CLI has more priority. 
    
    For instance, if the value of an argument has default value, and if the value of 
    that argument is defined in YAML file, the latter value will be kept. 

    If the value of an argument is specified manually/directly from the CLI (not default),
    but if the value of that argument is defined in YAML file, the latter value will be 
    ignored. Hence, the former value will be kept.
    """
    if args.from_config is not None:
        with open(args.from_config, 'r') as file:
            config_from_yaml = yaml.safe_load(file)

        for key, value in config_from_yaml.items():
            if hasattr(args, key):  
                current_value = getattr(args, key)
                default_value = parser.get_default(key)
                if current_value == default_value:
                    setattr(args, key, value)

        # return updated args
        return args
    else:
        # return args as is (unmodified)
        return args

def is_push_to_hf_hub_enabled(push_enabled, dataset_id, split):
    if push_enabled is True:
        if dataset_id is None or split is None:
            raise ValueError("push_to_hub was set to True, "
                             "but dataset id and split name are set to None")
        else:
            return True
