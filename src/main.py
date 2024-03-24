import argparse

from pipeline.eval import eval_on_records
from utils import get_steps_right

def main(args):
    valid, input_steps = get_steps_right(args.steps)
    if valid:
        if "ft" in input_steps:
            pass
        
        if "eval" in input_steps:
            qualification, (avg_similarity_scores, avg_precision_scores) = eval_on_records(
                args.ft_model_id, 
                args.test_ds_id, args.test_ds_split, 
                args.ft_model_config_path, args.prompt_tmpl_path,
                args.avg_similarity_threshold, args.avg_precision_threshold,
                args.gemini_api_key
            )
            print(f"qualification: {qualification}, avg_similarity_scores: {avg_similarity_scores}, avg_precision_scores: {avg_precision_scores}")
        
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
    
    # for eval step
    parser.add_argument("--test-ds-id", type=str, default=None)
    parser.add_argument("--test-ds-split", type=str, default="test_sft")
    parser.add_argument("--avg-similarity-threshold", type=float, default=90.0)
    parser.add_argument("--avg-precision-threshold", type=float, default=90.0)
    
    args = parser.parse_args()
    main(args)