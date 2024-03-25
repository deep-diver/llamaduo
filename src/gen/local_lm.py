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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        quantization_config=quantization_config, device_map="auto"
    )

    return tokenizer, model

def gen_model_output(model, tokenizer, ds, temperature=0.4, max_new_tokens=1024, delimiter="assistant\n"):
    """
    gen_model_output generates and return response(output) from a given model.

    arguments:
    model -- fine-tuned lanaguage model instance
    tokenizer -- tokenizer instance
    ds -- a single data record which has "prompt" column
    """
    messages = [
        {"role": "user", "content": ds['prompt']},
    ]

    gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    output_tensor = model.generate(
        gen_input,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    return tokenizer.decode(output_tensor[0], skip_special_tokens=True).split(delimiter)[1]