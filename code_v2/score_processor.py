import pandas as pd
import json
from pathlib import Path

if Path("./score.csv").exists():
    score_csv = pd.read_csv()
else:
    score_csv = pd.DataFrame()

for score_json_path in Path("./scores").glob("*.json"):
    score_json = json.load(open(score_json_path))
    print(score_json_path.name)
    print(score_json)

    score_dict = {}
    score_dict["name"] = "-".join(score_json_path.stem.split("_")[:2])
    score_dict["rouge-1"] = f"{score_json['rouge-1']['f']}"
    score_dict["rouge-2"] = f"{score_json['rouge-2']['f']}"
    score_dict["rouge-l"] = f"{score_json['rouge-l']['f']}"
    
    
    score_csv = score_csv.append(score_dict, ignore_index=True)
print(score_csv)
score_csv.to_csv("score.csv", index=False)