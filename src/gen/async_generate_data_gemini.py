import asyncio
import aiohttp
from typing import List, Dict
import os
import json
import numpy as np
from datasets import load_dataset


if os.getenv("COLAB_RELEASE_TAG"):
    from google.colab import userdata

    GOOGLE_API_KEY = userdata.get("GEMINI_ONE_API_KEY", None)
else:
    GOOGLE_API_KEY = os.getenv("GEMINI_ONE_API_KEY", None)

assert GOOGLE_API_KEY, "GOOGLE_API_KEY cannot be None."


MODEL_NAME = "gemini-1.0-ultra-latest"
NUM_DATASET_SAMPLES = 5
EXISTING_DATASET = load_dataset("sayakpaul/no_robots_only_coding", split="train_sft")
RATE_LIMIT_PER_MINUTE = 60
REQUEST_INTERVAL = 60 / RATE_LIMIT_PER_MINUTE  # Seconds between requests


# prompt template can be loaded from a `.toml` file.
def craft_prompt(instruction, response):
    prompt = """
Generate a series of (instruction, response) pairs that are similar in context and structure to the example provided below. Each pair should consist of a concise instruction followed by an appropriate, detailed response. The instruction should pose a clear task or question, while the response should provide a comprehensive answer or solution that could be understood by someone with a basic understanding of the subject.

Example pair:

Instruction: {instruction}
Response: {response}

Your task is to generate more pairs that maintain this level of clarity and detail. The topic is Coding. Ensure that the responses are informative and accurate, suitable for an educational context.

Store the generated pairs in JSON format, with each pair as an object within an array. Each object should have two key-value pairs: "instruction" and "response". For instance:

[{{"instruction": "text", "response": "text"}}, {{"instruction": "text", "response": "text"}}, ...]

Remember to maintain consistency in the format and ensure the generated pairs are diverse and cover a broad range of subjects. You must return the response
in the asked format and you must not add any additional text in your response.
"""
    return prompt.format(instruction=instruction, response=response)


async def generate_text(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GOOGLE_API_KEY}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                # rate-limiting
                await asyncio.sleep(REQUEST_INTERVAL)
                return result
            else:
                print(f"Error: {response.status}")
                return None


async def main():
    # craft prompts
    total_original_samples = len(EXISTING_DATASET)
    random_indices = np.random.randint(0, total_original_samples, size=(NUM_DATASET_SAMPLES))
    prompts = []
    print("Preparing candidate prompts.")
    for random_index in random_indices:
        sample = EXISTING_DATASET[int(random_index)]["messages"]
        instruction = sample[0]["content"]
        response = sample[1]["content"]
        prompt_for_sample = craft_prompt(instruction, response)
        prompts.append(prompt_for_sample)

    # create tasks
    tasks = []
    print("Creating tasks for asyncio.")
    for prompt in prompts:
        tasks.append(generate_text(prompt))

    # fire execution
    results = await asyncio.gather(*tasks)

    # collate results
    print("All tasks done. Collating results.")
    for i, result in enumerate(results):
        if result:
            with open(f"gemini_results_{i}.json", "w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    asyncio.run(main())
