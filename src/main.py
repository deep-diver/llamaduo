import argparse

from pipeline.eval import eval_on_records
from utils import validate_steps

def main(args):
    if args.load_in_8bit is True \
        and args.load_in_4bit is True:
        raise ValueError("both load_in_8bit and load_in_4bit are set. "
                         "only one of them should be set at a time")
    
    valid, input_steps = validate_steps(args.steps)
    if valid:
        if "fine-tuning" in input_steps:
            pass
        
        if "eval" in input_steps:
            print("processing eval step...")
            qualification_results = eval_on_records(
                args.ft_model_id,
                args.load_in_8bit, args.load_in_4bit,
                args.test_ds_id, args.test_ds_split, args.batch_size,
                args.ft_model_config_path, args.prompt_tmpl_path,
                args.eval_workers, args.avg_similarity_threshold, 
                args.avg_precision_threshold, args.gemini_api_key
            )
            print(qualification_results)
        
        if "synth-gen" in input_steps:
            pass
        
        if "deploy" in input_steps:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMOps pipeline CLI")
    parser.add_argument("--prompt-tmpl-path", type=str, default="config/prompts.toml")
    parser.add_argument("--gemini-api-key", type=str, default=None)
    parser.add_argument("--steps", type=str, nargs="+", help="List of pipeline steps to run")
    
    # common 
    parser.add_argument("--ft-model-id", type=str, default=None)
    parser.add_argument("--ft-model-config-path", type=str, default="config/sample_config.yaml")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    
    # for eval step
    parser.add_argument("--test-ds-id", type=str, default=None)
    parser.add_argument("--test-ds-split", type=str, default="test_sft")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--avg-similarity-threshold", type=float, default=90.0)
    parser.add_argument("--avg-precision-threshold", type=float, default=90.0)
    
    args = parser.parse_args()
    main(args)