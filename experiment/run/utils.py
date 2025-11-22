import json
import re

def record_json(inputs, output_path):
    with open(output_path, 'w', encoding = 'utf-8') as f:
        json.dump(inputs, f, ensure_ascii=False, indent = 4)

def read_json(output_path):
    with open(output_path, 'r', encoding = 'utf-8') as f:
        return json.load(f)
def parse_json_from_markdown(s: str):
    # ```json ... ``` 또는 ``` ... ``` 제거
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s.strip())
    return json.loads(s)