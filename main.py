import os
import asyncio
import argparse

import google.generativeai as genai

from src.pipeline.batch_inference import gen_local_lm_responses
from src.pipeline.eval import eval_on_records
from src.pipeline.utils import push_to_hf_hub

def _push_to_hf_hub_enabled(args):
    if args.push_lm_responses_to_hf_hub is True:
        if args.lm_response_dataset_id is None \
            or args.hf_token is None:
            raise ValueError("push_to_hub was set to True, but either or all of "
                            "lm_response_dataset_id and hf_token are set to None")
        else:
            return True    

def batch_inference(args):
    hf_hub = _push_to_hf_hub_enabled(args)

    if args.load_in_8bit is True \
        and args.load_in_4bit is True:
        raise ValueError("both load_in_8bit and load_in_4bit are set. "
                            "only one of them should be set at a time")

    local_lm_responses = gen_local_lm_responses(
        args.ft_model_id, args.load_in_8bit, args.load_in_4bit,
        args.test_ds_id, args.test_ds_split, 
        args.batch_infer_data_preprocess_bs, args.inference_bs, args.repeat,
        args.lm_response_dataset_split, args.ft_model_config_path, 
    )

    if hf_hub is True:
        push_to_hf_hub(
            args.lm_response_dataset_id, 
            args.lm_response_dataset_split, local_lm_responses,
            args.hf_token, args.lm_response_append
        )

    return local_lm_responses

async def evaluation(args):
    hf_hub = _push_to_hf_hub_enabled(args)

    eval_results = await eval_on_records(
        args.lm_response_dataset_id, args.lm_response_dataset_split,
        args.prompt_tmpl_path, args.service_model_name, args.eval_workers,
        args.avg_similarity_threshold, args.avg_precision_threshold,
        args.eval_data_preprocess_bs, args.eval_dataset_split
    )

    if hf_hub is True:
        push_to_hf_hub(
            args.eval_dataset_id, 
            args.eval_dataset_split, eval_results["ds_with_scores"], 
            args.hf_token, False
        )

    return eval_results

async def main(args):
    if args.gemini_api_key is not None:
        genai.configure(api_key=args.gemini_api_key)

    match args.step:
        case "fine-tune":
            pass

        case "batch-infer":
            print("batch inference step...")
            local_lm_responses = batch_inference(args)
            print(local_lm_responses)

        case "eval":
            print("processing eval step...")
            eval_results = await evaluation(args)
            print(eval_results)            

        case "synth-gen":
            pass

        case "deploy":
            pass

        case _:
            raise ValueError("step is ste to dis-allowed value. Choose one from "
                            "[fine-tune, batch-infer, eval, synth-gen, deploy]")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="LLMOps pipeline CLI")
    
    # global
    parser.add_argument("--prompt-tmpl-path", type=str, 
                        default=os.path.abspath("config/prompts.toml"),
                        help="Path to the prompts TOML configuration file.")
    parser.add_argument("--gemini-api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                        help="Gemini API key for authentication.")
    parser.add_argument("--service-model-name", type=str, default="gemini-1.0-pro",
                        help="Which service LLM to use for evaluation of the local fine-tuned model")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--step", type=str, default=None,
                        help="step to run in the choices of [fine-tune, batch-infer, eval, synth-gen, deploy].")
    
    # batch inference
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
    parser.add_argument("--lm-response-dataset-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    parser.add_argument("--lm-response-dataset-split", type=str, default="batch_infer",
                        help="Split of the lm response dataset to use for saving or retreiving.")
    parser.add_argument("--lm-response-append", action="store_true", default=True,
                        help="Wheter to overwrite or append on the existing Hugging Face Dataset repository")

    # eval
    parser.add_argument("--eval-data-preprocess-bs", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=4,
                        help="Number of workers to use for parallel evaluation.")
    parser.add_argument("--avg-similarity-threshold", type=float, default=90.0,
                        help="Average similarity threshold for passing evaluation.")
    parser.add_argument("--avg-precision-threshold", type=float, default=90.0,
                        help="Average precision threshold for passing evaluation.")
    parser.add_argument("--push-eval-to-hf-hub", action="store_true",
                        help="Whether to push generated evaluation to Hugging Face Dataset repository(Hub)")
    parser.add_argument("--eval-dataset-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    parser.add_argument("--eval-dataset-split", type=str, default="eval",
                        help="Split of the lm evak dataset to use for saving.")    
    args = parser.parse_args()    
    
    args = parser.parse_args()
    asyncio.run(main(args))