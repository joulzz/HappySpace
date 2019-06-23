import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast


def center_node(x):
    x = ast.literal_eval(x)
    x = (((int(x[0][0])+int(x[1][0]))/2),((int(x[0][1])+int(x[1][1]))/2))
    return x

output_df = pd.read_csv('2018-11-17.csv')
# Node- (x,y) center of the face bounding box
output_df['Node'] = pd.DataFrame(output_df['Last_Location'].values.tolist())
output_df['Node'] = output_df['Node'].apply(center_node)

# Distance- Euclidean Distance between two nodes
for i, (index, row) in enumerate(output_df.iterrows()):
    if i == len(output_df) - 1:
        break
    node_a = np.array(output_df.loc[index,'Node'])
    node_b = np.array(output_df.loc[index+1,'Node'])
    output_df.loc[index,'Distance'] = np.linalg.norm(node_a - node_b)
print(output_df)

moving_average = output_df['Distance'].mean()
print("Moving Average", moving_average)

range = (output_df['Distance'].max() - output_df['Distance'].min(skipna=True))
print("Range", range)

variance = output_df['Distance'].var(skipna=True)
print("Variance", variance)

# Unique ID Calibration

R = (1/100)*(range)
output_df.rename(columns={'ID': 'Reference ID'}, inplace=True)

for i, (index, row) in enumerate(output_df.iterrows()):
    if i == 0:
        output_df.loc[index, 'ID'] = output_df.loc[index, 'Reference ID']
        continue
    if output_df.loc[index,'Distance'] < R:
        output_df.loc[index,'ID']= output_df.loc[index-1,'Reference ID']
    else:
        output_df.loc[index, 'ID'] = output_df.loc[index,'Reference ID']

print(output_df)

# Unique Distributions Plot
output_df['Node'] = ast.literal_eval(output_df['Node'])
print(output_df['Node'])

# output_df.plot(kind='scatter',x='ID',y='Distance',color='red')
# plt.show()






