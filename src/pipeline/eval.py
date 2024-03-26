import toml
import asyncio
from string import Template
from datasets import load_dataset, concatenate_datasets

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
    for (instruction, messages, lm_response) in zip(
        batch_data['prompt'], batch_data['messages'], lm_responses
    ):
        ground_truth = messages[1]['content']
        
        eval_prompts.append(
            Template(eval_prompt_tmpl).substitute(
                instruction=instruction,
                human_response=ground_truth,
                lm_response=lm_response
            )         
        )

    return eval_prompts
    
def _get_test_dataset(dataset_id, split, tokenizer, batch_size):
    """
    _get_test_dataset returns the target test dataset
    """
    def __batch_process(batch):
        batch["input_ids"] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False
            )
            for prompt in batch["prompt"]
        ]
        return batch
        
    return load_dataset(dataset_id, split=split).map(
        __batch_process, batched=True, batch_size=batch_size
    )

async def _gen_eval_on_records(eval_prompts, eval_workers):
    """
    _gen_eval_on_records simultaneously generates evaluations on the eval_prompts
    """
    assessments = []
    for idx in range(0, len(eval_prompts), eval_workers):
        partial_eval_prompts = eval_prompts[idx:idx+eval_workers]
        tasks = [
            asyncio.create_task(
                call_service_llm(eval_prompt, retry_num=5)
            ) for eval_prompt in partial_eval_prompts
        ]
        results = await asyncio.gather(*tasks)
        assessments.extend(results)
        
    return assessments

async def _eval_on_records(model, tokenizer, batch_data, eval_prompt_tmpl, eval_workers):
    """
    _eval_on_single_record:
    1. generates output on a given instruction from data by local language model
    2. construct evaluation prompt by filling the generated output in
    3. evaluate the quality of the generated output via service language model (i.e. Gemini)
    """
    lm_responses = gen_model_outputs(model, tokenizer, batch_data)
    eval_prompts = _construct_eval_prompts(batch_data, lm_responses, eval_prompt_tmpl)
    assessments = await _gen_eval_on_records(eval_prompts, eval_workers)
    return assessments
    
async def eval_on_records(
    model_id, 
    load_in_8bit, load_in_4bit,
    test_dataset_id, test_dataset_split, 
    data_preprocess_bs, inference_bs, repeat,
    config_path, eval_prompt_tmpl_path,
    eval_workers, avg_similarity_threshold, avg_precision_threshold
):
    """
    eval_on_records evaluates the generated output on a given instruction dataset by local language model 
    """
    model_args, data_args, sft_args = get_args(config_path)
    tokenizer, model = get_model(load_in_8bit, load_in_4bit, model_args, data_args, sft_args, model_id=model_id)
    
    total_similarity_scores = 0
    total_precision_scores = 0
    ds = _get_test_dataset(test_dataset_id, test_dataset_split, tokenizer, data_preprocess_bs)
    eval_prompt_tmpl = _get_eval_prompt_tmpl(eval_prompt_tmpl_path)
    
    for idx in range(0, len(ds), inference_bs):
        for repeat_idx in range(repeat):
            batch_data = ds[idx:idx+inference_bs]
            assessments = await _eval_on_records(
                model, tokenizer, batch_data, eval_prompt_tmpl, eval_workers
            )
            
            for partial_idx, assessment in enumerate(assessments):
                similarity_score = assessment['similarity_assessment']['score']
                precision_score = assessment['precision_assessment']['score']
                total_similarity_scores = total_similarity_scores + similarity_score
                total_precision_scores = total_precision_scores + precision_score
                print(f"eval on (sample_num={idx+partial_idx}, repeat={repeat_idx}) / similarity_score: {similarity_score}, precision_score: {precision_score}")
    
    avg_similarity_scores = total_similarity_scores / (len(ds) * repeat)
    avg_precision_scores = total_precision_scores / (len(ds) * repeat)
    
    qualification = avg_similarity_scores >= avg_similarity_threshold and avg_precision_scores >= avg_precision_threshold
    return {
        "qualification": qualification, 
        "avg_similarity_scores": avg_similarity_scores,
        "avg_precision_scores": avg_precision_scores
    }