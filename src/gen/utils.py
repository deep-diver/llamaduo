import json
from .gemini import generate_async

def _find_json_snippet(raw_snippet):
	"""
	_find_json_snippet tries to find JSON snippets in a given raw_snippet string
	"""
	json_parsed_string = None

	json_start_index = raw_snippet.find('{')
	json_end_index = raw_snippet.rfind('}')

	if json_start_index >= 0 and json_end_index >= 0:
		json_snippet = raw_snippet[json_start_index:json_end_index+1]
		try:
			json_parsed_string = json.loads(json_snippet, strict=False)
		except:
			raise ValueError('......failed to parse string into JSON format')
	else:
		raise ValueError('......No JSON code snippet found in string.')

	return json_parsed_string

def _parse_first_json_snippet(snippet):
	"""
	_parse_first_json_snippet tries to find JSON snippet and parse into json object
	"""
	json_parsed_string = None

	if isinstance(snippet, list):
		for snippet_piece in snippet:
			try:
				json_parsed_string = _find_json_snippet(snippet_piece)
				return json_parsed_string
			except:
				pass
	else:
		try:
			json_parsed_string = _find_json_snippet(snippet)
		except Exception as e:
			raise ValueError(str(e))

	return json_parsed_string

def _required_keys_exist(json_dict, keys_to_check):
    """
    Checks if required keys (including nested keys) exist in the given JSON dictionary.
    """

    def check_nested_keys(data, keys):
        if not keys:  # Base case: All keys found
            return True
        current_key = keys[0]
        if current_key in data:
            return check_nested_keys(data[current_key], keys[1:])
        else:
            return False

    for key_path in keys_to_check:
        keys = key_path.split(".")  # Split nested keys like "a.b" into a list
        if not check_nested_keys(json_dict, keys):
            raise KeyError(f"Missing required keys: {key_path}")

    return json_dict  # If all checks pass, return the dictionary

async def call_service_llm(eval_model, prompt, keys_to_check, retry_num=10, job_num=None):
    """
    call_service_llm makes API call to service language model (currently Gemini)
    it makes sure the generated output by the service language model in a certain JSON format
    if no valid JSON format found, call_service_llm retries up to the number of retry_num
    """
    json_dict = None
    cur_retry = 0

    while json_dict is None and cur_retry < retry_num:
        try:
            assessment = await generate_async(
                eval_model, prompt=prompt,
            )

            json_dict = _parse_first_json_snippet(assessment)
            json_dict = _required_keys_exist(json_dict, keys_to_check)
        except KeyError as e:
            cur_retry = cur_retry + 1
            json_dict = None
            print(f"......retry [{e}]")			
        except Exception as e:
            cur_retry = cur_retry + 1
            print(f"......retry [{e}]")

    return job_num, json_dict


def _calculate_job_distribution(rate_limit_per_minute, num_workers):
    """
    Calculates how many jobs to launch simultaneously and the sleep interval
    to respect a given rate limit per minute with multiple concurrent workers.

    Args:
        rate_limit_per_minute (int): The maximum number of jobs allowed per minute.
        num_workers (int): The number of concurrent workers.

    Returns:
        tuple: (jobs_per_batch, sleep_seconds)
    """

    # Calculate the maximum number of jobs allowed per second
    jobs_per_second = rate_limit_per_minute / 60

    # Estimate how many jobs a single worker can handle in a second, assuming 20 sec per job
    jobs_per_worker_per_second = 1 / 20  

    # Calculate how many jobs to launch per batch to stay within the rate limit
    jobs_per_batch = int(jobs_per_second / jobs_per_worker_per_second / num_workers)

    # If no jobs can be launched per batch due to constraints, handle it gracefully
    if jobs_per_batch == 0:
        jobs_per_batch = 1  # Launch at least one job
        print("Warning: Rate limit and job duration may lead to exceeding the limit.")

    # Calculate the sleep time between batches
    sleep_seconds = 60 / jobs_per_batch

    return jobs_per_batch, sleep_seconds