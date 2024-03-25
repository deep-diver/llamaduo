import toml
import asyncio
from string import Template
from datasets import load_dataset

from gen.local_lm import get_model, gen_model_outputs
from gen.utils import call_service_llm
from pipeline.utils import get_args

def _get_eval_prompt_tmpl(eval_prompt_tmpl_path):
    """
    _get_eval_prompt_tmpl returns pre-defined prompt template for 
    evaluation of language model's generated output
    """
    prompts = toml.load(eval_prompt_tmpl_path)
    return prompts['eval']['prompt']

def _construct_eval_prompts(batch_data, lm_responses, eval_prompt_tmpl):
    """
    construct_eval_prompt returns a prompt to be injected into the language model (evaluator)

    arguments:
    ds -- a batch data records which has "prompt", "messages" columns
    lm_response -- string value which fine-tuned model generated
    eval_prompt_tmpl -- string with placeholders of instruction, human_response, and lm_response.
    """
    eval_prompts = []
    for (data, lm_response) in zip(batch_data, lm_responses):
        instruction = data['prompt']
        ground_truth = data['messages'][1]['content']
        
        eval_prompts.append(
            Template(eval_prompt_tmpl).substitute(
                instruction=instruction,
                human_response=ground_truth,
                lm_response=lm_response
            )         
        )

    return eval_prompts
    
def _get_test_dataset(dataset_id, split):
    """
    _get_test_dataset returns the target test dataset
    """
    return load_dataset(dataset_id)[split]

async def _gen_eval_on_records(eval_prompts, gemini_api_key, eval_workers):
    assessments = []
    for idx in range(0, len(eval_prompts), eval_workers):
        partial_eval_prompts = eval_prompts[idx:idx+eval_workers]
        tasks = [
            asyncio.create_task(
                call_service_llm(eval_prompt, gemini_api_key, retry_num=5)
            ) for eval_prompt in partial_eval_prompts
        ]
        results = await asyncio.gather(*tasks)
        assessments.extend(results)
        
    return assessments

async def _eval_on_records(model, tokenizer, batch_data, eval_prompt_tmpl, gemini_api_key, eval_workers):
    """
    _eval_on_single_record:
    1. generates output on a given instruction from data by local language model
    2. construct evaluation prompt by filling the generated output in
    3. evaluate the quality of the generated output via service language model (i.e. Gemini)
    """
    lm_responses = gen_model_outputs(model, tokenizer, batch_data)
    eval_prompts = _construct_eval_prompts(batch_data, lm_responses, eval_prompt_tmpl)
    assessments = await _gen_eval_on_records(eval_prompts, gemini_api_key, eval_workers)
    return assessments
    
def eval_on_records(
    model_id, 
    load_in_8bit, load_in_4bit,
    test_dataset_id, test_dataset_split, batch_size,
    config_path, eval_prompt_tmpl_path,
    eval_workers, avg_similarity_threshold, avg_precision_threshold, gemini_api_key
):
    """
    eval_on_records evaluates the generated output on a given instruction dataset by local language model 
    """
    model_args, data_args, _ = get_args(config_path)
    tokenizer, model = get_model(model_id, load_in_8bit, load_in_4bit, model_args, data_args)
    
    total_similarity_scores = 0
    total_precision_scores = 0
    ds = _get_test_dataset(test_dataset_id, test_dataset_split)
    eval_prompt_tmpl = _get_eval_prompt_tmpl(eval_prompt_tmpl_path)
    
    for idx in range(0, len(ds), batch_size):
        batch_data = ds[idx:idx+batch_size]
        assessments = _eval_on_records(model, tokenizer, batch_data, eval_prompt_tmpl, gemini_api_key, eval_workers)
        
        for partial_idx, assessment in enumerate(assessments):
            similarity_score = assessment['similarity_assessment']['score']
            precision_score = assessment['precision_assessment']['score']
            total_similarity_scores = total_similarity_scores + similarity_score
            total_precision_scores = total_precision_scores + precision_score
            print(f"eval on {idx+partial_idx}...similarity_score: {similarity_score}, precision_score: {precision_score}")
    
    avg_similarity_scores = total_similarity_scores / len(ds)
    avg_precision_scores = total_precision_scores / len(ds)
    
    qualification = avg_similarity_scores >= avg_similarity_threshold and avg_precision_scores >= avg_precision_threshold
    return {
        "qualification": qualification, 
        "avg_similarity_scores": avg_similarity_scores,
        "avg_precision_scores": avg_precision_scores
    }