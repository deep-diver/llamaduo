import os
import glob
import toml
import json
import asyncio
import numpy as np
from tqdm import tqdm
from string import Template
from typing import List, Dict

from datasets import load_dataset
from datasets import Dataset, DatasetDict

from ..gen.gemini import get_model as get_service_model
from ..gen.utils import call_service_llm

def _load_all_json_files(filenames):
    all_json_dicts = []
    for filename in filenames:
        with open(filename) as f:
            all_json_dicts.append(json.load(f))
    return all_json_dicts

def _format_response(responses: List[Dict[str, str]]):
    final_instruction_answer_pairs = []

    for response in responses["contents"]:
        user_response_dict = {}
        assistant_response_dict = {}
        user_response_dict["content"] = response["instruction"]
        user_response_dict["role"] = "user"
        assistant_response_dict["content"] = response["response"]
        assistant_response_dict["role"] = "assistant"

        final_instruction_answer_pairs.append([user_response_dict, assistant_response_dict])

    return final_instruction_answer_pairs

def _sampling(dataset_id, split, num_sample):
    samples = []
    ds = load_dataset(dataset_id, split=split)
    total_original_samples = len(ds)
    random_indices = np.random.randint(
        0, total_original_samples, size=(num_sample)
    ).tolist()
    for idx in random_indices:
        samples.append(ds[int(idx)])
    return samples

def _get_synth_data_gen_prompt_tmpl(prompt_tmpl_path):
    prompts = toml.load(prompt_tmpl_path)
    return prompts['synth_data_gen']['prompt']

def _craft_prompt(prompt_tmpl, instruction, response, topic):
    return Template(prompt_tmpl).substitute(
        instruction=instruction,
        response=response,
        topic=topic
    )

def _craft_prompts(samples, topic, prompt_tmpl_path):
    prompt_tmpl = _get_synth_data_gen_prompt_tmpl(prompt_tmpl_path)
    prompts = [
        _craft_prompt(
            prompt_tmpl,
            sample["messages"][0]["content"], 
            sample["messages"][1]["content"], 
            topic
        )
        for sample in samples
    ]
    return prompts

async def _gen_synth_data(prompts, model, eval_workers):
    """
    _gen_eval_on_records simultaneously generates evaluations on the eval_prompts
    """
    generated_data = []
    keys_to_check = ["contents"]
    for idx in tqdm(range(0, len(prompts), eval_workers), desc="batches"):
        partial_prompts = prompts[idx:idx+eval_workers]
        tasks = [
            asyncio.create_task(
                call_service_llm(model, eval_prompt, keys_to_check, retry_num=5, job_num=idx)
            ) for eval_prompt in partial_prompts
        ]
        results = await asyncio.gather(*tasks)
        results = sorted(results, key=lambda item: item[0])
        results = [result[1] for result in results]
        generated_data.extend(results)
        
    return generated_data

async def synth_data_generation(
    dataset_id, split, num_sample,
    topic, prompt_tmpl_path,
    service_model_name, gen_workers,
    save_dir_path
):
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    samples = _sampling(dataset_id, split, num_sample)
    prompts = _craft_prompts(samples, topic, prompt_tmpl_path)

    gen_model = get_service_model(service_model_name)
    generated_data = await _gen_synth_data(prompts, gen_model, gen_workers)

    filenames = []
    for i, data in tqdm(enumerate(generated_data), total=len(generated_data), desc="to JSON file"):
        if data:
            filename = f"{save_dir_path}/generated_data_{i}.json"
            filenames.append(filename)

            with open(filename, "w") as f:
                json.dump(data, f)

    return filenames

def collage_as_dataset(
    filenames, service_model_name, topic
):
    all_json_dicts = _load_all_json_files(filenames)

    all_formatted_responses = []
    for json_dict in all_json_dicts:
        formatted_responses = _format_response(json_dict)
        for formatted_response in formatted_responses:
            all_formatted_responses.append(formatted_response)
    
    prompts = [service_model_name] * len(all_formatted_responses)
    prompt_ids = ["gemini-generated"] * len(all_formatted_responses)
    categories = [topic] * len(all_formatted_responses)
    dataset_train = Dataset.from_dict(
        {"prompt": prompts, "prompt_id": prompt_ids, "messages": all_formatted_responses, "category": categories}
    )
    return DatasetDict(train_sft=dataset_train)