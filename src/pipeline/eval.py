import toml
import asyncio
from tqdm import tqdm
from string import Template
from datetime import datetime
from datasets import load_dataset, DatasetDict

from ..gen.gemini import get_model as get_service_model
from ..gen.utils import call_service_llm

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
        
    return load_dataset(dataset_id, split=split).map(
        __batch_process, batched=True, batch_size=batch_size
    )

async def _gen_eval_on_records(eval_prompts, eval_model, eval_workers):
    """
    _gen_eval_on_records simultaneously generates evaluations on the eval_prompts
    """
    assessments = []
    keys_to_check = {"similarity_assessment", "precision_assessment"}
    for idx in range(0, len(eval_prompts), eval_workers):
        partial_eval_prompts = eval_prompts[idx:idx+eval_workers]
        tasks = [
            asyncio.create_task(
                call_service_llm(eval_model, eval_prompt, keys_to_check, retry_num=5, job_num=idx)
            ) for eval_prompt in partial_eval_prompts
        ]
        results = await asyncio.gather(*tasks)
        results = sorted(results, key=lambda item: item[0])
        results = [result[1] for result in results]
        assessments.extend(results)
        
    return assessments
    
async def eval_on_records(
    lm_response_dataset_id, lm_response_dataset_split,
    eval_prompt_tmpl_path, service_model_name, eval_workers, 
    avg_similarity_threshold, avg_precision_threshold,
    batch_size, eval_dataset_split
):
    """
    eval_on_records evaluates the generated output on a given instruction dataset by local language model 
    """
    today = datetime.today().strftime("%Y-%m-%d")
    similarity_scores = []
    precision_scores = []

    total_similarity_score = 0
    total_precision_score = 0
        
    eval_model = get_service_model(service_model_name)
    eval_prompt_tmpl = _get_eval_prompt_tmpl(eval_prompt_tmpl_path)
    lm_response_ds = _get_lm_response_dataset(lm_response_dataset_id, lm_response_dataset_split, eval_prompt_tmpl, batch_size=batch_size)

    for idx in tqdm(range(0, len(lm_response_ds), eval_workers), desc="batches"):
        batch_data = lm_response_ds[idx:idx+eval_workers]
        assessments = await _gen_eval_on_records(batch_data["eval_prompts"], eval_model, eval_workers)

        for partial_idx, assessment in enumerate(assessments):
            similarity_score = assessment['similarity_assessment']['score']
            precision_score = assessment['precision_assessment']['score']
            similarity_scores.append(similarity_score)
            precision_scores.append(precision_score)

            total_similarity_score = total_similarity_score + similarity_score
            total_precision_score = total_precision_score + precision_score
            print(f"eval on (sample_num={idx+partial_idx}) / similarity_score: {similarity_score}, precision_score: {precision_score}")

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
