import json
from gen.gemini import call_gemini

def find_json_snippet(raw_snippet):
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

def parse_first_json_snippet(snippet):
	json_parsed_string = None

	if isinstance(snippet, list):
		for snippet_piece in snippet:
			try:
				json_parsed_string = find_json_snippet(snippet_piece)
				return json_parsed_string
			except:
				pass
	else:
		try:
			json_parsed_string = find_json_snippet(snippet)
		except Exception as e:
			print(e)
			raise ValueError()

	return json_parsed_string

def call_service_llm(prompt, gemini_api_key, retry_num=3):
    assessment_json = None
    cur_retry = 0

    while assessment_json is None and cur_retry < retry_num:
        try:
            assessment = call_gemini(
                prompt=prompt,
                api_key=gemini_api_key
            )

            assessment_json = parse_first_json_snippet(assessment)
        except Exception as e:
            cur_retry = cur_retry + 1
            print(f"......retry [{e}]")

    return assessment_json