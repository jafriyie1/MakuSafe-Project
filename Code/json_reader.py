import json
import os
import pandas as pd

def json_concat():
    path = "../Data/"
    files = []
    merged_files = ""
    for i in os.listdir(path):
        if i.endswith(".json"):
            print(i)
            with open(path+i) as json_data:
                d = json.load(json_data)
                files.append(d)

    with open(path+"merged_files.json", "w") as outfile:
        json.dump(files, outfile)

    with open(path+"merged_files.json", "r") as files:
        merged_files = json.load(files)

    return merged_files


def df_read_multiple_json():
    path = "../Data/"
    files = []
    data = ""
    for i in os.listdir(path):
        if i.endswith(".json"):
            print(i)
            with open(path+i) as json_data:
                data = pd.DataFrame(json.loads(line) for line in json_data)
                files.append(data)

    df = pd.concat(files)
    return df
