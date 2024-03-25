import toml
from string import Template
from datasets import load_dataset

from gen.local_lm import get_model, gen_model_output
from gen.utils import call_service_llm
from pipeline.utils import get_args

def _get_eval_prompt_tmpl(eval_prompt_tmpl_path):
    """
    _get_eval_prompt_tmpl returns pre-defined prompt template for 
    evaluation of language model's generated output
    """
    prompts = toml.load(eval_prompt_tmpl_path)
    return prompts['eval']['prompt']

def _construct_eval_prompt(ds, lm_response, eval_prompt_tmpl):
    """
    construct_eval_prompt returns a prompt to be injected into the language model (evaluator)

    arguments:
    ds -- a single data record which has "prompt", "messages" columns
    lm_response -- string value which fine-tuned model generated
    eval_prompt_tmpl -- string with placeholders of instruction, human_response, and lm_response.
    """
    instruction = ds['prompt']
    ground_truth = ds['messages'][1]['content']

    return Template(eval_prompt_tmpl).substitute(
        instruction=instruction,
        human_response=ground_truth,
        lm_response=lm_response
    )
    
def _get_test_dataset(dataset_id, split):
    """
    _get_test_dataset returns the target test dataset
    """
    return load_dataset(dataset_id)[split]

def _eval_on_single_record(model, tokenizer, data, eval_prompt_tmpl, gemini_api_key):
    """
    _eval_on_single_record:
    1. generates output on a given instruction from data by local language model
    2. construct evaluation prompt by filling the generated output in
    3. evaluate the quality of the generated output via service language model (i.e. Gemini)
    """
    lm_response = gen_model_output(model, tokenizer, data)
    eval_prompt = _construct_eval_prompt(data, lm_response, eval_prompt_tmpl)
    assessments = call_service_llm(eval_prompt, gemini_api_key, retry_num=5)
    return assessments

def eval_on_records(
    model_id, 
    load_in_8bit, load_in_4bit,
    test_dataset_id, test_dataset_split,
    config_path, eval_prompt_tmpl_path,
    avg_similarity_threshold, avg_precision_threshold, gemini_api_key
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
    
    for idx, data in enumerate(ds):
        assessments = _eval_on_single_record(model, tokenizer, data, eval_prompt_tmpl, gemini_api_key)
        similarity_score = assessments['similarity_assessment']['score']
        precision_score = assessments['precision_assessment']['score']
        total_similarity_scores = total_similarity_scores + similarity_score
        total_precision_scores = total_precision_scores + precision_score
        print(f"eval on {idx}...similarity_score: {similarity_score}, precision_score: {precision_score}")
    
    avg_similarity_scores = total_similarity_scores / len(ds)
    avg_precision_scores = total_precision_scores / len(ds)
    
    qualification = avg_similarity_scores >= avg_similarity_threshold and avg_precision_scores >= avg_precision_threshold
    return {
        "qualification": qualification, 
        "avg_similarity_scores": avg_similarity_scores,
        "avg_precision_scores": avg_precision_scores
    }