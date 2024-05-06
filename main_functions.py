import pandas as pd
import json
import re
from openai import OpenAI

def extract_categories(survey_question, 
                       survey_input,
                       user_model,
                       creativity,
                       categories,
                       api_key):
    
    client = OpenAI(api_key=api_key)
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    cat_num = len(categories)
    category_dict = {str(i+1): 0 for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)
    
    link1 = []
    extracted_jsons = []
    
    for response in survey_input:
        prompt = f"""A survey respondent was asked, "{survey_question}" \
        Their response is here in triple backticks: ```{response}```. \
        First, thoruoughly extract all their answers to the question and be as specific as possible. \
        Second, take these reasons and select all of the following numbered categories they fall into: \
        "{categories_str}" \
        Third, form your response in proper JSON format. \
        The number belonging to the category shoulbe be the key and a 1 is the key value if the category is present. \
        If none of the categories are present in their response, provide 0's for all key values in your JSON."""
        print(prompt)
        try:
            response = client.chat.completions.create(
                model=user_model,
                messages=[
                    {
                      "role": "system",
                      "content": f"""You are an expert in identifying themes and patterns in open-ended survey responses to the question, "{survey_question}". \
                      When given a survey response, you analyze it critically and thoroughly to identify user-provided categories present in the response."""
                    },
                    {'role': 'user', 
                     'content': prompt}
                ],
                temperature=creativity
            )

            reply = response.choices[0].message.content
            link1.append(reply)
            
            extracted_json = re.findall(r'```json\n(\{.*?\})\n```', reply, re.DOTALL)
            
            if extracted_json:
                cleaned_json = extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
                extracted_jsons.append(cleaned_json)
                print(cleaned_json)
            else:
                error_message = """{"1":"e"}"""
                extracted_jsons.append(error_message)
                print(error_message)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            link1.append(f"Error processing input: {response}")
            extracted_jsons.append(f"Error processing input: {response}")
            
    normalized_data_list = []
    
    for json_str in extracted_jsons:
        try:
            parsed_obj = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed_obj))
        except json.JSONDecodeError:
            default_json_obj = {"1": "j"}
            normalized_data_list.append(pd.json_normalize(default_json_obj))

    normalized_data = pd.concat(normalized_data_list, ignore_index=True)
    
    categorized_data = pd.DataFrame()
    categorized_data['survey_response'] = survey_input
    categorized_data['link1'] = link1
    categorized_data['json'] = extracted_jsons
    
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)
    
    return categorized_data
