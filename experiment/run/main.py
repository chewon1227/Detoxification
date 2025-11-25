import converse as c
import os
import sys
import json
import utils
import rag
from dotenv import load_dotenv

load_dotenv()
base_model_name = os.getenv("BASE_MODEL_NAME")
detox_model_name = os.getenv("DETOX_MODEL_NAME")

def main():
    # input preparation
    n = int(sys.argv[1])
    mode_num = int(sys.argv[2])
    init_persona_name = sys.argv[3]
    target_persona_name = sys.argv[4]
    init_persona_path = f"experiment/data/personas/{init_persona_name}.json"
    target_persona_path = f"experiment/data/personas/{target_persona_name}.json"

    init_persona = utils.read_json(init_persona_path)
    target_persona = utils.read_json(target_persona_path)

    # get database client
    client = rag.database_check()
    
    # set result path
    mn = ""
    if(base_model_name == "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
        mn = "naver-hyperclovax-seed-1_5B"
    if(base_model_name == "/tmp/Qwen2.5-14B-Instruct-bnb-4bit"):
        mn = "Qwen2.5-14B-Instruct"

    os.makedirs(f"experiment/result/{mn}/{init_persona['topic']}", exist_ok = True)

    # base(toxic) mode
    if (mode_num==0): 
        base_trajectory = c.agent_chat(n, init_persona, target_persona, mode="base", client = client)
        base_trajectory_path = f"experiment/result/{mn}/{init_persona['topic']}/base_trajectory_{n}turn_3.json"
        utils.record_json(base_trajectory, base_trajectory_path)
    # detox mode
    elif (mode_num ==1):
        detox_trajectory = c.agent_chat(n, init_persona, target_persona, mode = "detox", client = client)
        detox_trajectory_path = f"experiment/result/{mn}/{init_persona['topic']}/detox_trajectory_{n}turn_3.json"
        utils.record_json(detox_trajectory, detox_trajectory_path)
    # toxic vs detox mode
    elif (mode_num == 2):
        bad_good_trajectory = c.agent_chat(n, init_persona, target_persona, mode = "bad_good", client = client)
        bad_good_trajectory_path = f"experiment/result/{mn}/{init_persona['topic']}/bad_good_trajectory_{n}turn_2.json"
        utils.record_json(bad_good_trajectory, bad_good_trajectory_path)

if __name__ == "__main__":
    main()