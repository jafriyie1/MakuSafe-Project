import pandas as pd
import re

df = pd.read_csv("incidents_test.csv")

column_size = len(df.columns) - 1



#print((df.columns))
#print(df["raw"])
print()
#print(df)
json = []


def convert_string(string):
    print(string)
    string.strip('\\')
    return string

df['raw'] = df['raw'].str.replace('\\', '')
df['raw'] = df['raw'].str.replace('\"', '')
print(df["raw"])
print(df.iloc[12:column_size])
print(df.columns[12])

df["accel_data"] = df.iloc[12:column_size].apply(lambda x: "".join(x),axis=1)
print(df["accel_data"])

output = ""
for j in range(len(df)):
    for i in range(12, column_size):
        temp  = df.iloc[j:,i]
        temp = str(temp.values)
        #print(temp)
        output = "".join((output, temp))
        output = output
        #temp = {"accel_data": df.iloc[1:,12:column_size]
    output = output + "}"
print(output)




'''
df["raw"] = df.raw.apply(str)
df["raw"] = df[["raw"]].apply(convert_string)
print(df["raw"])
'''


#print(df[["raw"]])
#print(df.iloc[:,12])


#print(df["raw"])
#print(df.iloc[:, 12])




#df.loc[(df.iloc[:,10] == 0) & df['other_column'].isin(some_values)]
