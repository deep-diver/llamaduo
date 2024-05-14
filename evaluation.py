import os
import asyncio
import argparse

from genai_apis import APIFactory

from utils import is_push_to_hf_hub_enabled, update_args
from src.pipeline.eval import eval_on_records
from src.pipeline.utils import push_to_hf_hub

async def evaluate(args):
    """
    evaluation generates evaluations on given pairs of target and candidate responses.
    its main job is to call eval_on_records() function.
    Additionally it goes through arguments' validation, and 
    it pushes generated evaluations to the specified Hugging Face Dataset repo.
    """
    service_llm_kwargs = {
        "api_key": args.service_llm_api_key,
        "GCP_PROJECT_ID": args.gcp_project_id,
        "GCP_PROJECT_LOCATION": args.gcp_location,
        "AWS_LOCATION": args.aws_location,
    }
    service_llm_client = APIFactory.get_api_client(args.service_llm_provider, **service_llm_kwargs)

    hf_hub = is_push_to_hf_hub_enabled(
        args.push_eval_to_hf_hub,
        args.eval_ds_id, args.eval_ds_split
    )
    eval_results = await eval_on_records(
        args.lm_response_ds_id, args.lm_response_ds_split, args.prompt_tmpl_path, 
        service_llm_client, args.service_model_name,
        args.eval_workers, args.eval_repeat,
        args.avg_similarity_threshold, args.avg_precision_threshold,
        args.eval_data_preprocess_bs, args.eval_ds_split, args.rate_limit_per_minute
    )

    if hf_hub is True:
        # dataset with columns of (instructions, target_response, candidate_response
        # eval_prompts, similarity_scores, precision_scores) will be recorded
        push_to_hf_hub(
            args.eval_ds_id, args.eval_ds_split, 
            eval_results["ds_with_scores"], False
        )
    else:
        eval_results["ds_with_scores"].save_to_disk(args.eval_ds_id)

    print(eval_results)
    return eval_results

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="CLI for batch inference step")

    parser.add_argument("--service-llm-provider", type=str, default="gemini",
                        help="Which service LLM provider to choose")
    parser.add_argument("--service-llm-api-key", type=str, default=os.getenv("SERVICE_LLM_API_KEY"),
                        help="API KEY for selected service LLM. Credentials for GCP, AWS based LLM, "
                        "use dedicated authentication CLI (ignore this option)")
    parser.add_argument("--service-model-name", type=str, default="gemini-1.0-pro",
                        help="Which service LLM to use for evaluation of the local fine-tuned model")
    parser.add_argument("--gcp-project-id", type=str, default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--gcp-location", type=str, default=os.getenv("GCP_LOCATION"))
    parser.add_argument("--aws-location", type=str, default=os.getenv("AWS_LOCATION"))

    parser.add_argument("--from-config", type=str, default="config/evaluation.yaml",
                        help="set CLI options from YAML config")

    parser.add_argument("--rate-limit-per-minute", type=int, default=60,
                        help="Rate-limit per minute for the service LLM.")
    parser.add_argument("--prompt-tmpl-path", type=str, 
                        default=os.path.abspath("config/prompts.toml"),
                        help="Path to the prompts TOML configuration file.")
    parser.add_argument("--lm-response-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    parser.add_argument("--lm-response-ds-split", type=str, default="batch_infer",
                        help="Split of the lm response dataset to use for saving or retreiving.")
    parser.add_argument("--eval-data-preprocess-bs", type=int, default=16)
    parser.add_argument("--eval-repeat", type=int, default=10)
    parser.add_argument("--eval-workers", type=int, default=4,
                        help="Number of workers to use for parallel evaluation.")
    parser.add_argument("--avg-similarity-threshold", type=float, default=90.0,
                        help="Average similarity threshold for passing evaluation.")
    parser.add_argument("--avg-precision-threshold", type=float, default=90.0,
                        help="Average precision threshold for passing evaluation.")
    parser.add_argument("--push-eval-to-hf-hub", action="store_true",
                        help="Whether to push generated evaluation to Hugging Face Dataset repository(Hub)")
    parser.add_argument("--eval-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    parser.add_argument("--eval-ds-split", type=str, default="eval",
                        help="Split of the lm evak dataset to use for saving.") 
    args = parser.parse_args()
    args = update_args(parser, args)

    asyncio.run(evaluate(args))