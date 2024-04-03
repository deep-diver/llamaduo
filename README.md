# LLMOps Pipeline

This project showcase LLMOps pipeline that fine-tune a small size LLM model to prepare the outage of the service LLM. For this project, we choose [Gemini 1.0 Pro](https://deepmind.google/technologies/gemini/) for service type LLM and [Gemma](https://blog.google/technology/developers/gemma-open-models/) 2B/7B for small sized LLM model.

For this project, the following tech stacks are chosen:
- Hugging Face open source ecosystem ([transformers](https://github.com/huggingface/transformers), [peft](https://github.com/huggingface/peft), [alignment-handbook](https://github.com/huggingface/alignment-handbook), [Hub](https://huggingface.co/docs/hub/en/index))
- [Gemini API](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini) on [Google AI Studio](https://ai.google.dev/).

## Motivation

We assume that small size LLM could show comparable performance to that of service type LLM on a specific task, and this project is to prove such possibility. Furthermore, this project shows how to smoothly migrate from service type LLM to small sized LLM when 
- we experience the outage of service type LLM which could cause disasters on many service/applications that rely on the service type LLM.
- we want decide to use small sized LLM hosted on local servers for the cost savings or privacy issues.
- ......

## Overview

This project comes with the toolset of batch inference, evaluation, and synthetic data generation. Each tool can be run independently, but they could be hooked up to form a pipeline. It is your choice how to use them.

The prerequisite to run these toolset is to have a dataset consisting of desired `(prompt, response)` pairs. The exact format of the dataset could be found [here](https://huggingface.co/datasets/sayakpaul/no_robots_only_coding). The `prompt` is the input to the small size LLM to generate output. Then, `prompt`, `response`, and the `generated output` are going to be used to evaluate the fine-tuned small size LLM. The main idea is to make small size LLM to output as much as similar to the given response.

### Batch inference

Batch inference lets fine-tuned LLM to generate text and push the results on the Hugging Face Dataset repository. 

To perform this you need to run the following commands in terminal:

```console
# HF_TOKEN is required to access gated model repository 
# and push the resulting outputs to the Hugging Face Hub.
$ export HF_TOKEN=<YOUR-HUGGINGFACE-ACCESS-TOKEN>

# All parameters defined in the config/batch_inference.yaml file
# could be manually inputted as CLI arguments (arg names are the same)
$ python batch_inference.py --from-config config/batch_inference.yaml
```

Then, the resulting outputs will be pushed to Hugging Face Dataset repository in the following structure:

| column names | instructions |  target_responses |  candidate_responses  | model_id  |  model_sha |
|---|---|---|---|---|---|
| descriptions | the input prompts | desired outputs |  model generated outputs  |  model id that generated outputs  |  the version of the model |

### Evaluation

Evaluation evaluates the generated text from fine-tuned LLM with the help of service LLM. The evaluation criteria is the similarity and quality by comparing to the given desired outputs.

(Instruction WIP)

### Synthetic data generation

Synthetic data generation generates similar data to the ones used to fine-tune the LLM. This could be performed based on the evaluation results. For instance, if you are not satisfied with the evaluation results, and if you think the training dataset is not large enough, you can create more of the similar data to boost the performance of the LLM.

(Instruction WIP)

## Acknowledgments

This is a project built during the Gemma/Gemini sprints held by Google's ML Developer Programs team. We are thankful to be granted good amount of GCP credits to finish up this project.