import utils
import json
import run_model as m
import rag
import os
from dotenv import load_dotenv                                    

def generate_one_utterance(tokenizer,
                           model,
                           init_persona, 
                           target_persona,  
                           context,
                           client):

    x = m.run_model_generate_chat_utt(tokenizer, model,init_persona, target_persona, context, client)

    # try:
    #     output = json.loads(x)
    # except:
    #     output = x

    print("\n")
    print("=================final utterance:===================")
    init_persona_name = init_persona["name"]
    print(f"{init_persona_name}: {x}")

    return x


def agent_chat(n, init_persona, target_persona, mode:str, client):

    load_dotenv()
    QUESTION = os.getenv("QUESTION")

    # load base model
    init_tokenizer, init_model = rag.model_setup(mode)
    # target_tokenizer, target_model = rag.model_setup(mode)
    target_tokenizer = init_tokenizer
    target_model = init_model

    # data preparation
    history = []
    history.append({
        "meta": {
            "topic": init_persona["topic"],
            "mode": mode,
            "Question": QUESTION,
            init_persona["name"] : init_persona,
            target_persona["name"]: target_persona,

        }
    })

    context = f"{QUESTION}\n"

    # run a conversation
    for i in range(n):
        turn_data = {}
        turn_data["turn"] = i+1
        #init's turn
        utt= generate_one_utterance(init_tokenizer, init_model, init_persona, target_persona, context, client)

        turn_data[init_persona["name"]] = utt
        context = f"{init_persona['name']}: {utt}\n" # context = init_persona
        
        #target's turn
        utt= generate_one_utterance(target_tokenizer, target_model,target_persona, init_persona, context, client)
        
        turn_data[target_persona["name"]] = utt
        context = f"{target_persona['name']}: {utt}\n"  # context = target_persona

        # save
        history.append(turn_data)

    return history
    