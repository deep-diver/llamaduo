import json
import google.generativeai as genai

async def call_gemini(prompt="", api_key=None, generation_config=None, safety_settings=None):
    genai.configure(api_key=api_key)
    
    if generation_config is None:
        generation_config = {
          "temperature": 0.9,
          "top_p": 1,
          "top_k": 32,
          "max_output_tokens": 8192,
        }

    if safety_settings is None:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]

    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    prompt_parts = [prompt]
    response = await model.generate_content_async(prompt_parts)
    return response.text