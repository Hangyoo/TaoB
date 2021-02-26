import pandas as pd
data_tmp = pd.read_csv('dataset.csv')
data = data_tmp.values.tolist()
for i, data in enumerate(data):
    print(i)