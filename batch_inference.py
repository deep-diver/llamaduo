import os
import yaml
import argparse

from utils import is_push_to_hf_hub_enabled
from src.pipeline.batch_inference import gen_local_lm_responses
from src.pipeline.utils import push_to_hf_hub

def batch_inference(args):
    """
    batch_inference generates outputs on given instruction.
    its main job is to call gen_local_lm_responses() function.
    Additionally it goes through arguments' validation, and 
    it pushes generated outputs to the specified Hugging Face Dataset repo.
    """
    hf_hub = is_push_to_hf_hub_enabled(
        args.push_lm_responses_to_hf_hub,
        args.lm_response_ds_id, args.lm_response_ds_split, args.hf_token
    )

    if args.load_in_8bit is True \
        and args.load_in_4bit is True:
        raise ValueError("both load_in_8bit and load_in_4bit are set. "
                            "only one of them should be set at a time")

    local_lm_responses = gen_local_lm_responses(
        args.ft_model_id, args.load_in_8bit, args.load_in_4bit,
        args.test_ds_id, args.test_ds_split, 
        args.batch_infer_data_preprocess_bs, args.inference_bs, args.repeat,
        args.lm_response_ds_split, args.ft_model_config_path, 
    )

    if hf_hub is True:
        # dataset with columns of 
        # (instructions, target_response, candidate_response) will recorded
        push_to_hf_hub(
            args.lm_response_ds_id, args.lm_response_ds_split, 
            local_lm_responses, 
            args.hf_token, args.lm_response_append
        )

    return local_lm_responses

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
            if hasattr(args, key):  # Check if the attribute exists
                current_value = getattr(args, key)
                default_value = parser.get_default(key)
                if current_value == default_value:
                    setattr(args, key, value)

    return args

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="CLI for batch inference step")

    parser.add_argument("--gemini-api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                        help="Gemini API key for authentication.")
    parser.add_argument("--service-model-name", type=str, default="gemini-1.0-pro",
                        help="Which service LLM to use for evaluation of the local fine-tuned model")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--from-config", type=str, default="config/batch_inference.yaml",
                        help="set CLI options from YAML config")

    parser.add_argument("--ft-model-id", type=str, default=None,
                        help="ID of the fine-tuned model to use.")
    parser.add_argument("--ft-model-config-path", type=str, 
                        default=os.path.abspath("config/sample_config.yaml"),
                        help="Path to the fine-tuned model configuration file.")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load the model weights in 8-bit quantization.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load the model weights in 4-bit quantization.")    
    parser.add_argument("--test-ds-id", type=str, default=None,
                        help="ID of the test dataset.")
    parser.add_argument("--test-ds-split", type=str, default="test_sft",
                        help="Split of the test dataset to use (e.g., 'test_sft').")
    parser.add_argument("--batch-infer-data-preprocess-bs", type=int, default=16,
                        help="Batch size for data preprocessing.")
    parser.add_argument("--inference-bs", type=int, default=4,
                        help="Batch size for model inference.")
    parser.add_argument("--repeat", type=int, default=4,
                        help="Number of times to repeat the evaluation for each data sample")
    parser.add_argument("--push-lm-responses-to-hf-hub", action="store_true",
                        help="Whether to push generated responses to Hugging Face Dataset repository(Hub)")
    parser.add_argument("--lm-response-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    parser.add_argument("--lm-response-ds-split", type=str, default="batch_infer",
                        help="Split of the lm response dataset to use for saving or retreiving.")
    parser.add_argument("--lm-response-append", action="store_true", default=True,
                        help="Wheter to overwrite or append on the existing Hugging Face Dataset repository")
    args = parser.parse_args()
    args = update_args(parser, args)
    print(args)

    # batch_inference(args)