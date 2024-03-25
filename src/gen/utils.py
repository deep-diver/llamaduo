import json
from gen.gemini import call_gemini

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

def _required_keys_exist(assessment_json):
    """
    _required_keys_exist checks if required keys exist in the given assessment_json
    """    
    keys_to_check = set(["precision_assessment", "precision_assessment"])
    qualified = keys_to_check.issubset(set(assessment_json.keys()))
    if qualified is False:
        raise ValueError("missing required keys")

async def call_service_llm(prompt, gemini_api_key, retry_num=3):
    """
    call_service_llm makes API call to service language model (currently Gemini)
    it makes sure the generated output by the service language model in a certain JSON format
    if no valid JSON format found, call_service_llm retries up to the number of retry_num
    """
    assessment_json = None
    cur_retry = 0

    while assessment_json is None and cur_retry < retry_num:
        try:
            assessment = await call_gemini(
                prompt=prompt,
                api_key=gemini_api_key
            )

            assessment_json = _parse_first_json_snippet(assessment)
            _required_keys_exist(assessment_json)
        except Exception as e:
            cur_retry = cur_retry + 1
            print(f"......retry [{e}]")

    return assessment_json