import pandas as pd

f = open('incidents.csv', 'r')
data_string = f.read()
data_string[:300]

thing_str = "\\" + "\""
print(thing_str)

f.close()

data_string2 = data_string.replace(thing_str, "")
data_string2[:300]

o = open("fixed_incidents.csv", "w")
o.write(data_string2)
o.close()
f.close()

data = pd.read_csv("fixed_incidents.csv")
data.head()

data['RawType'] = 0
possible_types = ['accel', 'temp', 'humidity', 'light', 'manual', 'sound']
row_type_list = ['Unknown' for i in range(data.shape[0])]
for row in data.itertuples():
    for label in possible_types:
        if label in row.raw:
            row_type_list[row.Index] = label
            break
data['RawType'] = row_type_list

row_value_list = [None for i in range(data.shape[0])]
for row in data.itertuples():
    val = row.raw.split(': ')[1].replace("}", "")
    row_value_list[row.Index] = val

data['RawValue'] = row_value_list

numeric_vals = data[data['RawType'] != 'accel']
numeric_vals['RawValue'] = numeric_vals['RawValue'].astype(float)
numeric_vals['AccelValue'] = None

accel_data = data[data['RawType'] == 'accel']
accel_data['AccelValue'] = accel_data['RawValue'].copy()
accel_data['RawValue'] = -1

new_data = pd.concat([accel_data, numeric_vals])

new_data.sort_index(inplace=True)
print(new_data['RawValue'])

new_data.to_csv("incidents_more_cols.csv", index=False)
