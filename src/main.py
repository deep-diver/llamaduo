import asyncio
import argparse

import google.generativeai as genai

from pipeline.eval import eval_on_records
from utils import validate_steps

async def main(args):
    if args.gemini_api_key is not None:
        genai.configure(api_key=args.gemini_api_key)
    
    if args.load_in_8bit is True \
        and args.load_in_4bit is True:
        raise ValueError("both load_in_8bit and load_in_4bit are set. "
                         "only one of them should be set at a time")
    
    valid, input_steps = validate_steps(args.steps)
    if valid is True:
        if "fine-tuning" in input_steps:
            pass
        
        if "eval" in input_steps:
            print("processing eval step...")
            qualification_results = await eval_on_records(
                args.ft_model_id,
                args.load_in_8bit, args.load_in_4bit,
                args.test_ds_id, args.test_ds_split, 
                args.data_preprocess_bs, args.inference_bs, args.repeat,
                args.ft_model_config_path, args.prompt_tmpl_path,
                args.eval_workers, args.avg_similarity_threshold, 
                args.avg_precision_threshold, 
            )
            print(qualification_results)
        
        if "synth-gen" in input_steps:
            pass
        
        if "deploy" in input_steps:
            pass

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="LLMOps pipeline CLI")
    
    # global
    parser.add_argument("--prompt-tmpl-path", type=str, default="config/prompts.toml",
                        help="Path to the prompts TOML configuration file.")
    parser.add_argument("--gemini-api-key", type=str, default=None,
                        help="Gemini API key for authentication.")
    parser.add_argument("--steps", type=str, nargs="+",
                        help="List of pipeline steps to run in the choices of [fine-tune, eval, synth-gen, deploy].")
    
    # common
    parser.add_argument("--ft-model-id", type=str, default=None,
                        help="ID of the fine-tuned model to use.")
    parser.add_argument("--ft-model-config-path", type=str, default="config/sample_config.yaml",
                        help="Path to the fine-tuned model configuration file.")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load the model weights in 8-bit quantization.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load the model weights in 4-bit quantization.")
    
    # eval
    parser.add_argument("--test-ds-id", type=str, default=None,
                        help="ID of the test dataset.")
    parser.add_argument("--test-ds-split", type=str, default="test_sft",
                        help="Split of the test dataset to use (e.g., 'test_sft').")
    parser.add_argument("--data-preprocess-bs", type=int, default=16,
                        help="Batch size for data preprocessing.")
    parser.add_argument("--inference-bs", type=int, default=4,
                        help="Batch size for model inference.")
    parser.add_argument("--repeat", type=int, default=4,
                        help="Number of times to repeat the evaluation for each data sample")
    parser.add_argument("--eval-workers", type=int, default=4,
                        help="Number of workers to use for parallel evaluation.")
    parser.add_argument("--avg-similarity-threshold", type=float, default=90.0,
                        help="Average similarity threshold for passing evaluation.")
    parser.add_argument("--avg-precision-threshold", type=float, default=90.0,
                        help="Average precision threshold for passing evaluation.")
    args = parser.parse_args()    
    
    args = parser.parse_args()
    asyncio.run(main(args))