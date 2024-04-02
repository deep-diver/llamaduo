def is_push_to_hf_hub_enabled(push_enabled, dataset_id, split, hf_token):
    if push_enabled is True:
        if dataset_id is None or split is None or hf_token is None:
            raise ValueError("push_to_hub was set to True, but either or all of "
                            "lm_response_dataset_id and hf_token are set to None")
        else:
            return True
