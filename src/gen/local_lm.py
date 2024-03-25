import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from alignment.model_utils import get_tokenizer

def get_model(model_id, load_in_8bit, load_in_4bit, model_args, data_args):
    """
    get_model instantiates and return fine-tuned language model and tokenzier.

    arguments:
    model_args -- ModelArguments obtained from H4ArgumentParser
    data_args -- DataArguments obtained from H4ArgumentParser
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.padding_size = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        quantization_config=quantization_config, device_map="auto"
    )

    return tokenizer, model

def gen_model_outputs(model, tokenizer, ds, temperature=0.4, max_new_tokens=1024, delimiter="assistant\n"):
    """
    gen_model_output generates and return response(output) from a given model.

    arguments:
    model -- fine-tuned lanaguage model instance
    tokenizer -- tokenizer instance
    ds -- a batch data records which has "prompt" column
    """
    messages = []
    for data in ds:
        messages.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": data['prompt']}],
                add_generation_prompt=True, tokenize=False
            )
        )

    input_ids = tokenizer.batch_encode_plus(
        input_ids, 
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    generated_ids = model.generate(
        **input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    
    outputs = []
    raw_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for raw_output in raw_outputs:
        outputs.append(raw_output.split(delimiter)[1])

    return outputs