import datasets
from datasets import load_dataset

from ..gen.local_lm import get_model, gen_model_outputs
from ..pipeline.utils import get_args

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

def gen_local_lm_responses(
    model_id, load_in_8bit, load_in_4bit,
    test_dataset_id, test_dataset_split, 
    data_preprocess_bs, inference_bs, repeat,
    lm_response_dataset_split, config_path, 
):
    model_args, data_args, sft_args = get_args(config_path)
    tokenizer, model = get_model(load_in_8bit, load_in_4bit, model_args, data_args, sft_args, model_id=model_id)
    ds = _get_test_dataset(test_dataset_id, test_dataset_split, tokenizer, data_preprocess_bs)

    results = {"instructions": [], "target_responses": [], "candidate_responses": []}

    for idx in range(0, len(ds), inference_bs):
        for repeat_idx in range(repeat):
            batch_data = ds[idx:idx+inference_bs]
            lm_responses = gen_model_outputs(model, tokenizer, batch_data)

            for messages, lm_response in zip(batch_data["messages"], lm_responses):
                instruction = messages[0]["content"]
                target_response = messages[1]["content"]

                results["instructions"].append(instruction)
                results["target_responses"].append(target_response)
                results["candidate_responses"].append(lm_response)

    return datasets.Dataset.from_dict(
        results, split=lm_response_dataset_split
    )