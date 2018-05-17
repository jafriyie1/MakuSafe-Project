import json
import os
import pandas as pd
import gc

def json_concat():
    path = "../Data/CollectedDataJSONs/"
    files = []
    merged_files = ""

    open(path+"merged_files.json", "w").close()

    for i in os.listdir(path):
        if i.endswith(".json"):
            print(i)
            if i != "merged_files.json":
                with open(path+i) as json_data:
                    d = json.load(json_data)
                    files.append((d))

    #d = pd.DataFrame(json.load(json_data))

    list_df = [pd.DataFrame(d) for d in files]

    new_list = []
    #print(list_df)

    for n in range(len(list_df)):
        new_list.append(list())

    #print(len(new_list))
    #print(new_list[0])

    #new_list = []
    total_list = []
    for count, df in enumerate(list_df):
        #print(df)
        #print
        result = ""
        for i in range(len(df)):
            #print(df.iloc[i].values[0])
            t = df.iloc[i].values[0]
            key = t.keys()

            test = ' '.join(t.keys())
            result = ''.join([i for i in test if not i.isdigit()])

            t[result] = t[test]
            del t[test]

            #print(new_list[0])
            new_list[count].append(t)
            #print(new_list)
        #print(result)
        d = pd.DataFrame(new_list[count], dtype= "float64")
        d = d.rename(columns={result: 'data'})
        #print(d.columns)
        #d = d.dropna()
        d["class"] = count
        #print(count)
        #print(d)
        #d = pd.DataFrame()
        #print(d)
        total_list.append(d)
        gc.collect()
        del d
    '''
    with open(path+"merged_files.json", "w") as outfile:
        json.dump(files, outfile)


    with open(path+"merged_files.json", "r") as files:
        merged_files = json.load(files)
    '''
    #print(total_list)
    return total_list


def df_read_multiple_json():
    path = "../Data/CollectedDataJSONs"
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



def df_reads_specific_json():
    path = "../Data/"
    df = pd.read_json(path+"upstairs.json")
    df_two = pd.read_json(path+"downstairs.json")

    full_df = pd.concat([df, df_two])

    #print(full_df)
    return full_df


if __name__ == "__main__":
    json_concat()
