import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from alignment.model_utils import get_tokenizer

def get_model(load_in_8bit, load_in_4bit, model_args, data_args, sft_args, model_id=None):
    """
    get_model instantiates and return fine-tuned language model and tokenzier.

    arguments:
    model_args -- ModelArguments obtained from H4ArgumentParser
    data_args -- DataArguments obtained from H4ArgumentParser
    """
    tokenizer = get_tokenizer(model_args, data_args)
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    
    model = AutoModelForCausalLM.from_pretrained(
        sft_args.hub_model_id if model_id is None else model_id, 
        quantization_config=quantization_config, 
        torch_dtype=torch.bfloat16, device_map="auto"
    )

    return tokenizer, model

def gen_model_outputs(model, tokenizer, batch_data, temperature=0.4, max_new_tokens=1024, delimiter="assistant\n"):
    """
    gen_model_output generates and return response(output) from a given model.

    arguments:
    model -- fine-tuned lanaguage model instance
    tokenizer -- tokenizer instance
    ds -- a batch data records which has "prompt" column
    """
    input_ids = tokenizer(
        batch_data["input_ids"], 
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(
        **input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5
    )
    
    outputs = []
    raw_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for raw_output in raw_outputs:
        outputs.append(raw_output.split(delimiter)[1])

    return outputs