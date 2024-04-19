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
    """
    synth_data_gen generates synthetic dataset that look similar to the original dataset.
    its main job is to call synth_data_generation() and collage_as_dataset() in order.

    synth_data_generation() generates and save synthetic dataset in external JSON files,
    then collage_as_dataset() re-organize into Hugging Face Dataset object.

    Additionally it goes through arguments' validation, and 
    it pushes generated evaluations to the specified Hugging Face Dataset repo.
    """    
    if args.gemini_api_key is not None:
        genai.configure(api_key=args.gemini_api_key)

    hf_hub = is_push_to_hf_hub_enabled(
        args.push_synth_ds_to_hf_hub,
        args.synth_ds_id, args.synth_ds_split
    )
    filenames = await synth_data_generation(
        args.reference_ds_id, args.reference_ds_split, 
        args.seed, args.num_samples,
        args.topic, args.prompt_tmpl_path,
        args.service_model_name, args.gen_workers, args.rate_limit_per_minute
    )
    dataset = collage_as_dataset(
        filenames, args.service_model_name, args.topic, args.synth_ds_split
    )

    if hf_hub is True:
        # dataset with columns of (prompts, prompt_ids, messages, categories) will be recorded
        push_to_hf_hub(
            args.synth_ds_id, args.synth_ds_split, 
            dataset, args.synth_ds_append
        )
    else:
        dataset.save_to_disk(args.synth_ds_id)
    
    for filename in filenames:
        os.remove(filename)

    return dataset

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="CLI for synthetic generation step")

    parser.add_argument("--gemini-api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                        help="Gemini API key for authentication.")

    parser.add_argument("--from-config", type=str, default="config/synth_data_gen.yaml",
                        help="set CLI options from YAML config")

    parser.add_argument("--service-model-name", type=str, default="gemini-1.0-pro",
                        help="Which service LLM to use for evaluation of the local fine-tuned model")
    parser.add_argument("--rate-limit-per-minute", type=int, default=60,
                        help="Rate-limit per minute for the service LLM.")
    parser.add_argument("--prompt-tmpl-path", type=str, 
                        default=os.path.abspath("config/prompts.toml"),
                        help="Path to the prompts TOML configuration file")
    parser.add_argument("--reference-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID of the original dataset to sample")
    parser.add_argument("--reference-ds-split", type=str, default=None,
                        help="Split of the reference dataset")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="How many data to sample from the reference dataset")
    parser.add_argument("--seed", type=int, default=2004,
                        help="Seed to generate indicies for the samples")
    parser.add_argument("--topic", type=str, default=None,
                        help="What kind of topics/tasks that synthetic dataset will cover")
    parser.add_argument("--gen-workers", type=int, default=4,
                        help="How many workers to process synthetic dataset generation")
    parser.add_argument("--push-synth-ds-to-hf-hub", action="store_true",
                        help="Whether to push generated synthetic data to Hugging Face Dataset repository(Hub)")
    parser.add_argument("--synth-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID that synthetic dataset to be pushed")
    parser.add_argument("--synth-ds-split", type=str, default="eval",
                        help="Split of the synthetic dataset") 
    parser.add_argument("--synth-ds-append", action="store_true", default=True,
                        help="Wheter to overwrite or append on the existing Hugging Face Dataset repository")

    args = parser.parse_args()
    args = update_args(parser, args)

    asyncio.run(synth_data_gen(args))