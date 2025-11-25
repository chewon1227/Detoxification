import sys
from pathlib import Path

import rag as rag

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.append(str(TRAIN_ROOT))

from train.src.train.chat_prompt import build_converse_prompt  # noqa: E402


def run_model_generate_chat_utt(tokenizer,
                                model,
                                init_persona, 
                                target_persona, 
                                context,
                                query,
                                client):
    summary = build_converse_prompt(init_persona, target_persona, context)
    query = f"{init_persona['stance']}?"
    output = rag.generate_rag_response_local(tokenizer, model, client, query, summary)

    return output['answer']


    # def create_prompt_input(init_persona, target_persona, context):
    #     persona = init_persona

    #     persona_str = ""
    #     for key, val in persona.items():
    #         persona_str += f"{key}: {val}\n"
        
    #     target_persona_str = ""
    #     for key, val in target_persona.items():
    #         target_persona_str += f"{key}: {val}\n"

    #     # curr_chat_str = ""
    #     # for i in curr_chat:
    #     #     curr_chat_str += ": ".join(i) + "\n"
    #     # if curr_chat_str == "":
    #     #     curr_chat_str = "The conversation has not started yet -- start it!]"

    #     init_description = f"{persona_str}"
    #     target_description = f"{target_persona_str}"
    #     prompt_input = [
    #         context,          # 0 
    #         init_description,       # 1
    #         target_description,     # 2
    #         init_persona["name"],   # 3
    #         init_persona["topic"],  # 4
    #         target_persona["name"], # 5
    #         init_persona["name"],   # 6             
    #         init_persona["name"],   # 7 
    #         target_persona["name"], # 8
    #         init_persona["name"],   # 9
    #         init_persona["name"]]   # 10
    #     return prompt_input
    
    # def __chat_func_clean_up(gpt_response, prompt=""): 
    #     gpt_response = extract_first_json_dict(gpt_response)

    #     cleaned_dict = dict()
    #     cleaned = []
    #     for key, val in gpt_response.items(): 
    #         cleaned += [val]
    #     cleaned_dict["utterance"] = cleaned[0]
    #     cleaned_dict["end"] = True
    #     if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
    #         cleaned_dict["end"] = False

    #     return cleaned_dict

    # def __chat_func_validate(gpt_response, prompt=""): 
    #     print ("ugh...")
    #     try: 
    #     # print ("debug 1")
    #     # print (gpt_response)
    #     # print ("debug 2")

    #         print (extract_first_json_dict(gpt_response))
    #         # print ("debug 3")
    #         return True
    #     except:
    #         return False 
        
    # def get_fail_safe():
    #     cleaned_dict = dict()
    #     cleaned_dict["utterance"] = "..."
    #     cleaned_dict["end"] = False
    #     return cleaned_dict

    # print("11")
    # prompt_template = "YAICON_Model_fight/data/prompt/conversation.txt"
    # prompt_input = create_prompt_input(init_persona, target_persona, context)
    
    # print("22")
    # prompt = generate_prompt(prompt_input, prompt_template)
    
    # print(prompt)

#     # output = ms.get_model_request(prompt)

# def generate_prompt(curr_input, prompt_lib_file): 
#     if type(curr_input) == type("string"): 
#         curr_input = [curr_input]
#     curr_input = [str(i) for i in curr_input]

#     f = open(prompt_lib_file, "r", encoding = 'utf-8')
#     prompt = f.read()
#     f.close()
#     for count, i in enumerate(curr_input):   
#         prompt = prompt.replace(f"!<INPUT {count}>!", i)
#     if "<commentblockmarker>###</commentblockmarker>" in prompt: 
#         prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
#     return prompt.strip()

# def extract_first_json_dict(data_str):
# # Find the first occurrence of a JSON object within the string
#     start_idx = data_str.find('{')
#     end_idx = data_str.find('}', start_idx) + 1

#     # Check if both start and end indices were found
#     if start_idx == -1 or end_idx == 0:
#         return None

#     # Extract the first JSON dictionary
#     json_str = data_str[start_idx:end_idx]

#     try:
#         # Attempt to parse the JSON data
#         json_dict = json.loads(json_str)
#         return json_dict
#     except json.JSONDecodeError:
#         # If parsing fails, return None
#         return None
