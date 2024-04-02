from typing import List, Dict
from datasets import Dataset, DatasetDict
import glob
import json
import argparse


def load_all_json_files(path):
    all_json_files = glob.glob(f"{path}/*json")
    all_json_dicts = []
    for json_file in all_json_files:
        with open(json_file) as f:
            all_json_dicts.append(json.load(f))
    return all_json_dicts


def find_json_snippet(raw_snippet):
    """
    find_json_snippet tries to find JSON snippets in a given raw_snippet string
    """
    json_parsed_string = None

    json_start_index = raw_snippet.find("[")
    json_end_index = raw_snippet.rfind("]")

    if json_start_index >= 0 and json_end_index >= 0:
        json_snippet = raw_snippet[json_start_index : json_end_index + 1]
        try:
            json_parsed_string = json.loads(json_snippet, strict=False)
        except:
            raise ValueError("......failed to parse string into JSON format")
    else:
        raise ValueError("......No JSON code snippet found in string.")

    return json_parsed_string


def format_response(responses: List[Dict[str, str]]):
    final_instruction_answer_pairs = []

    for response in responses:
        # Sometimes `eval()` works sometimes `find_json_snippet()` works. I don't have a robust way
        # to handle this yet. Should contact Googlers or someone else.
        try:
            response = eval(response["candidates"][0]["content"]["parts"][0]["text"])
        except:
            try:
                response = find_json_snippet(response["candidates"][0]["content"]["parts"][0]["text"])
            except:
                print("JSON couldn't be parsed.")
                continue
        for res in response:
            user_response_dict = {}
            assistant_response_dict = {}
            user_response_dict["content"] = res["instruction"]
            user_response_dict["role"] = "user"
            assistant_response_dict["content"] = res["response"]
            assistant_response_dict["role"] = "assistant"

            final_instruction_answer_pairs.append([user_response_dict, assistant_response_dict])

    return final_instruction_answer_pairs


def main(args):
    print("Loading JSONs.")
    all_json_dicts = load_all_json_files(args.path)

    print("Formatting responses.")
    all_formatted_responses = []
    for json_dict in all_json_dicts:
        formatted_responses = format_response(json_dict)
        for formatted_response in formatted_responses:
            all_formatted_responses.append(formatted_response)

    print("Creating dataset.")
    prompts = ["gemini-generated"] * len(all_formatted_responses)
    prompt_ids = ["gemini-generated"] * len(all_formatted_responses)
    categories = ["Coding"] * len(all_formatted_responses)
    dataset_train = Dataset.from_dict(
        {"prompt": prompts, "prompt_id": prompt_ids, "messages": all_formatted_responses, "category": categories}
    )
    ds = DatasetDict(train_sft=dataset_train)
    print("Dataset successfully created.")
    print(ds)

    if args.dataset_hub_id is not None:
        ds.push_to_hub(args.dataset_hub_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", type=str)
    parser.add_argument("--dataset_hub_id", default=None, type=str)
    args = parser.parse_args()

    main(args)
