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
    n = int(sys.argv[1])
    init_persona_name = sys.argv[2]
    target_persona_name = sys.argv[3]
    init_persona_path = f"experiment/data/personas/{init_persona_name}.json"
    target_persona_path = f"experiment/data/personas/{target_persona_name}.json"

    init_persona = utils.read_json(init_persona_path)
    target_persona = utils.read_json(target_persona_path)

    # get database client
    client = rag.database_check()

    
    base_trajectory = c.agent_chat(n, init_persona, target_persona, mode="base", client = client)
    detox_trajectory = c.agent_chat(n, init_persona, target_persona, mode = "detox", client = client)
    
    
    if(base_model_name == "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
        mn = "naver-hyperclovax-seed-1_5B"

    os.makedirs(f"experiment/result/{mn}/{init_persona["topic"]}", exist_ok = True)
    base_trajectory_path = f"experiment/result/{mn}/{init_persona["topic"]}/base_trajectory_{n}turn.json"
    detox_trajectory_path = f"experiment/result/{mn}/{init_persona["topic"]}/detox_trajectory_{n}turn.json"
    utils.record_json(base_trajectory, base_trajectory_path)
    utils.record_json(detox_trajectory, detox_trajectory_path)


if __name__ == "__main__":
    main()