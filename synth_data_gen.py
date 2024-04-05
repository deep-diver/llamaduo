import os
import argparse
import asyncio
import google.generativeai as genai

from utils import is_push_to_hf_hub_enabled, update_args
from src.pipeline.synth_data_gen import (
    synth_data_generation, collage_as_dataset
)
from src.pipeline.utils import push_to_hf_hub

async def synth_data_gen(args):
    if args.gemini_api_key is not None:
        genai.configure(api_key=args.gemini_api_key)

    hf_hub = is_push_to_hf_hub_enabled(
        args.push_synth_ds_to_hf_hub,
        args.synth_ds_id, args.synth_ds_split, args.hf_token
    )
    filenames = await synth_data_generation(
        args.reference_ds_id, args.reference_ds_split, args.num_samples,
        args.topic, args.prompt_tmpl_path,
        args.service_model_name, args.gen_workers,
        args.save_dir_path
    )
    ds = collage_as_dataset(filenames, args.service_model_name, args.topic)

    if hf_hub is True:
        # dataset with columns of (instructions, target_response, candidate_response
        # eval_prompts, similarity_scores, precision_scores) will be recorded
        push_to_hf_hub(
            args.synth_ds_id, args.synth_ds_split, 
            ds, args.hf_token, False
        )
    
    for filename in filenames:
        os.remove(filename)

    return ds

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="CLI for synthetic generation step")

    parser.add_argument("--gemini-api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                        help="Gemini API key for authentication.")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--from-config", type=str, default="config/synth_data_gen.yaml",
                        help="set CLI options from YAML config")

    parser.add_argument("--service-model-name", type=str, default="gemini-1.0-pro",
                        help="Which service LLM to use for evaluation of the local fine-tuned model")
    parser.add_argument("--prompt-tmpl-path", type=str, 
                        default=os.path.abspath("config/prompts.toml"),
                        help="Path to the prompts TOML configuration file.")
    parser.add_argument("--reference-ds-id", type=str, default=None)
    parser.add_argument("--reference-ds-split", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--gen-workers", type=int, default=4)
    parser.add_argument("--save_dir_path", type=str, default="tmp")
    parser.add_argument("--push-synth-ds-to-hf-hub", action="store_true",
                        help="Whether to push generated evaluation to Hugging Face Dataset repository(Hub)")
    parser.add_argument("--synth-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    parser.add_argument("--synth-ds-split", type=str, default="eval",
                        help="Split of the lm evak dataset to use for saving.") 

    args = parser.parse_args()
    args = update_args(parser, args)

    asyncio.run(synth_data_gen(args))