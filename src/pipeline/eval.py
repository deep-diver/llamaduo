import toml
import asyncio
from tqdm import tqdm
from string import Template
from datetime import datetime
from datasets import load_dataset, DatasetDict
from collections import deque

from ..gen.gemini import get_model as get_service_model
from ..gen.utils import call_service_llm, _calculate_job_distribution

JSON_KEYS_TO_CHECK = ["similarity_assessment.score", "precision_assessment.score"]

def _get_eval_prompt_tmpl(eval_prompt_tmpl_path):
    """
    _get_eval_prompt_tmpl returns pre-defined prompt template for 
    evaluation of language model's generated output
    """
    prompts = toml.load(eval_prompt_tmpl_path)
    return prompts['eval']['prompt']

def _get_lm_response_dataset(dataset_id, split, eval_prompt_tmpl, batch_size):
    """
    _get_test_dataset returns the target test dataset
    """
    def __batch_process(batch):
        eval_prompts = []
        for (instruction, target_response, candidate_response) in zip(
            batch['instructions'], batch['target_responses'], batch['candidate_responses']
        ):
            eval_prompts.append(
                Template(eval_prompt_tmpl).substitute(
                    instruction=instruction,
                    human_response=target_response,
                    lm_response=candidate_response
                )         
            )
        batch["eval_prompts"] = eval_prompts
        return batch
        
    return load_dataset(dataset_id, split=split, verification_mode="no_checks").map(
        __batch_process, batched=True, batch_size=batch_size
    )

async def _gen_eval_on_records(eval_prompts, client, eval_model, gen_configs, eval_workers, rate_limit_per_minute):
    """
    _gen_eval_on_records simultaneously generates evaluations on the eval_prompts,
    respecting rate limits and scheduling constraints.
    """
    assessments = []
    jobs_at_once, sleep_interval = _calculate_job_distribution(rate_limit_per_minute, num_workers=eval_workers)
    prompt_queue = deque(eval_prompts)  # Use a deque to efficiently manage the queue of prompts

    while prompt_queue:
        tasks = []
        for _ in range(min(jobs_at_once, len(prompt_queue))):
            eval_prompt = prompt_queue.popleft()  # Take the prompt from the front of the queue
            task = asyncio.create_task(
                call_service_llm(client, eval_model, eval_prompt, gen_configs, JSON_KEYS_TO_CHECK, retry_num=10, job_num=len(assessments))
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        results = sorted(results, key=lambda item: item[0])
        assessments.extend(result[1] for result in results)  # Simplify processing and appending

        # Implement rate limiting
        await asyncio.sleep(sleep_interval)

    return assessments

def _iterate_inner_lists(outer_list):
    num_items = len(outer_list[0])  # Get number of items from any inner list
    for i in range(num_items):
        yield tuple(inner_list[i] for inner_list in outer_list)

async def eval_on_records(
    lm_response_dataset_id, lm_response_dataset_split, eval_prompt_tmpl_path, 
    service_llm_client, service_model_name, service_llm_gen_configs, eval_workers, eval_repeat,
    avg_similarity_threshold, avg_precision_threshold,
    batch_size, eval_dataset_split, rate_limit_per_minute
):
    """
    eval_on_records evaluates the generated output on a given instruction dataset by local language model 
    """
    today = datetime.today().strftime("%Y-%m-%d")
    similarity_scores = []
    precision_scores = []

    total_similarity_score = 0
    total_precision_score = 0
        
    # eval_model = get_service_model(service_model_name)

    eval_prompt_tmpl = _get_eval_prompt_tmpl(eval_prompt_tmpl_path)
    lm_response_ds = _get_lm_response_dataset(lm_response_dataset_id, lm_response_dataset_split, eval_prompt_tmpl, batch_size=batch_size)

    for idx in tqdm(range(0, len(lm_response_ds), eval_workers), desc="batches"):
        batch_data = lm_response_ds[idx:idx+eval_workers]

        partial_assessments = []
        for _ in tqdm(range(eval_repeat), desc="repeat"):        
            assessments = await _gen_eval_on_records(batch_data["eval_prompts"], service_llm_client, service_model_name, service_llm_gen_configs, eval_workers, rate_limit_per_minute)
            partial_assessments.append(assessments)

        for partial_idx, each_assessments in enumerate(_iterate_inner_lists(partial_assessments)):
            each_similarity_scores = 0
            each_precision_scores = 0

            for each_assessment in each_assessments:
                each_similarity_scores += int(each_assessment['similarity_assessment']['score'])
                each_precision_scores += int(each_assessment['precision_assessment']['score'])

            each_avg_similarity_score = each_similarity_scores / eval_repeat
            each_avg_precision_score = each_precision_scores / eval_repeat

            similarity_scores.append(each_avg_similarity_score)
            precision_scores.append(each_avg_precision_score)

            total_similarity_score = total_similarity_score + each_avg_similarity_score
            total_precision_score = total_precision_score + each_avg_precision_score

            print(f"eval on (sample_num={idx+partial_idx}) / similarity_score: {each_avg_similarity_score}, precision_score: {each_avg_precision_score}")     

    ds_with_scores = lm_response_ds.add_column("similarity_scores", similarity_scores)
    ds_with_scores = ds_with_scores.add_column("precision_scores", precision_scores)

    ds_with_scores = ds_with_scores.add_column("evaluators", [service_model_name]*len(similarity_scores))
    ds_with_scores = ds_with_scores.add_column("dates", [today]*len(similarity_scores))

    ds_with_scores = DatasetDict(
        {eval_dataset_split: ds_with_scores}
    )

    avg_similarity_scores = total_similarity_score / len(lm_response_ds)
    avg_precision_scores = total_precision_score / len(lm_response_ds)
    qualification = avg_similarity_scores >= avg_similarity_threshold and avg_precision_scores >= avg_precision_threshold

    return {
        "qualification": qualification,
        "avg_similarity_scores": avg_similarity_scores,
        "avg_precision_scores": avg_precision_scores,
        "ds_with_scores": ds_with_scores
    }
