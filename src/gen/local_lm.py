import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from alignment.model_utils import (
    get_tokenizer,
    get_quantization_config,
    get_kbit_device_map
)

def get_model(model_id, model_revision, model_args, data_args, sft_args):
    """
    get_model instantiates and return fine-tuned language model and tokenzier.

    arguments:
    model_args -- ModelArguments obtained from H4ArgumentParser
    data_args -- DataArguments obtained from H4ArgumentParser
    """
    model_id = sft_args.hub_model_id if model_id is None else model_id
    tokenizer = get_tokenizer(model_args, data_args)
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )    
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return tokenizer, model_id, model

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
    )
    
    outputs = []
    raw_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for raw_output in raw_outputs:
        outputs.append(raw_output.split(delimiter)[1])

    return outputs