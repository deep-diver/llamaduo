# LLaMADuo

<img src="assets/logo.png" style="display: block; margin-left: auto; margin-right: auto;">

This project showcase LLMOps pipeline that fine-tune a small size LLM model to prepare the outage of the service LLM. For this project, we choose [Gemini 1.0 Pro](https://deepmind.google/technologies/gemini/) for service type LLM and [Gemma](https://blog.google/technology/developers/gemma-open-models/) 2B/7B for small sized LLM model.

For this project, the following tech stacks are chosen:
- Hugging Face open source ecosystem ([`transformers`](https://github.com/huggingface/transformers), [`peft`](https://github.com/huggingface/peft), [`alignment-handbook`](https://github.com/huggingface/alignment-handbook), [`huggingface_hub`](https://huggingface.co/docs/hub/en/index))
- [Gemini API](https://ai.google.dev/docs).

## Motivation

We assume that a small LLM could show comparable performance to that of a service-type LLM on a specific task, and this project tries to showcase such a possibility in a practically grounded manner. Furthermore, this project shows how to smoothly migrate from service LLM to small LLM. 

Assume that service LLM is integrated into your service or system. However, from time to time, the service LLM should be replaced for the following reaons:
- failure of service LLM which may be operationally impacting for a business.
- data privacy issue. You don't want to expose your private data.
- some system runs without internet connection. Service LLM did a great job on PoC, but now you need the same intelligence in an on-premise environment.
- version control issue. Service LLMs changes their versions from time to time, and legacy versions will become obsolete. However, we just want to keep the behaviour as is.
- ...

To better prepare for such impacting situations, this project suggests to migrate from service LLM to a local small LLM. Since we are satisfied with the results from service LLM, we know our inputs (prompts) and the desired outputs. Then, we can fine-tune small size LLM on the collected prompts to match the desired outputs. Furthermore, if the fine-tuned LLM's performance is still poor, we can grow the size of the dataset by generating more of similar data via service LLM. 

## Overview

<img src="assets/figure.png" style="display: block; margin-left: auto; margin-right: auto;">

This project comes with the toolset of batch inference, evaluation, and synthetic data generation. Each tool can be run independently, but they could be hooked up to form a pipeline. It's on the end user to figure out the best way to collate these together. 

The prerequisite to run these toolset is to have a dataset consisting of desired `(prompt, response)` pairs. The exact format of the dataset could be found [here](https://huggingface.co/datasets/sayakpaul/no_robots_only_coding). The `prompt` is the input to the small size LLM to generate output. Then, `prompt`, `response`, and the `generated output` are going to be used to evaluate the fine-tuned small size LLM. The main idea is to make small size LLM to output as much as similar to the given response.

### Fine-tuning

We leverage Hugging Face's [alignment-handbook](https://github.com/huggingface/alignment-handbook) to streamline the LLM fine-tuning. Specifically, all the detailed fine-tuning parameters for this project could be found in [this config](config/sample_config.yaml). Also note that the same config can be reused for the batch inference in the next section to make sure there is no mimatched configurations.

Also, we are planning to add scripts to run the fine-tuning on the cloud. The list of supported cloud platform will be updated below: 
- [`dstack Sky`](https://sky.dstack.ai/): detailed instruction can be found in [dstack directory](dstack/).

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

Then, the resulting outputs will be pushed to Hugging Face Dataset repository in the following structure ([example](https://huggingface.co/datasets/chansung/lm_response_test)):

| column names | instructions |  target_responses |  candidate_responses  | model_id  |  model_sha |
|---|---|---|---|---|---|
| descriptions | the input prompts | desired outputs |  model generated outputs  |  model id that generated outputs  |  the version of the model |

### Evaluation

Evaluation evaluates the generated text from fine-tuned LLM with the help of service LLM. The evaluation criteria is the similarity and quality by comparing to the given desired outputs.

To perform this you need to run the following commands in terminal:

```console
# HF_TOKEN is required to access gated model repository 
# and push the resulting outputs to the Hugging Face Hub.
$ export HF_TOKEN=<YOUR-HUGGINGFACE-ACCESS-TOKEN>

# GEMINI_API_KEY is required to call Gemini API
$ export GEMINI_API_KEY=<YOUR-GEMINI-API-KEY>

# All parameters defined in the config/evaluation.yaml file
# could be manually inputted as CLI arguments (arg names are the same)
$ python batch_inference.py --from-config config/evaluation.yaml
```

Then, the resulting outputs will be pushed to Hugging Face Dataset repository in the following structure ([example](https://huggingface.co/datasets/chansung/eval_dataset_test)):

| column names | ommited.. | eval_prompts |  similarity_scores  | precision_scores  |  evaluators | dates |
|---|---|---|---|---|---|---|
| descriptions | all columns are copied from batch inference | prompts input to the evaluator | similarity score in 0~100 scale | precision score in 0~100 scale | model name used as evaluator | dates |

### Synthetic data generation

Synthetic data generation generates similar data to the ones used to fine-tune the LLM. This could be performed based on the evaluation results. For instance, if you are not satisfied with the evaluation results, and if you think the training dataset is not large enough, you can create more of the similar data to boost the performance of the LLM.

To perform this you need to run the following commands in terminal:

```console
# HF_TOKEN is required to access gated model repository 
# and push the resulting outputs to the Hugging Face Hub.
$ export HF_TOKEN=<YOUR-HUGGINGFACE-ACCESS-TOKEN>

# GEMINI_API_KEY is required to call Gemini API
$ export GEMINI_API_KEY=<YOUR-GEMINI-API-KEY>

# All parameters defined in the config/synth_data_gen.yaml file
# could be manually inputted as CLI arguments (arg names are the same)
$ python data_gen.py --from-config config/synth_data_gen.yaml
```

Then, the resulting outputs will be pushed to Hugging Face Dataset repository in the following structure ([example](https://huggingface.co/datasets/chansung/synth_ds_test2)):

| column names | generators | prompt_ids |  seed_prompts  | messages  |  category | 
|---|---|---|---|---|---|
| descriptions | model used to generate data | -- | the base prompts used to generate data | generated synthetic data | category this data belongs to |
## Acknowledgments

This is a project built during the Gemma/Gemini sprints held by Google's ML Developer Programs team. We are thankful to be granted good amount of GCP credits to finish up this project. Thanks to Hugging Face for providing Sayak with resources to run some fine-tuning experiments. 