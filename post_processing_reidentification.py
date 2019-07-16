import pandas as pd
import numpy as np

def cos_similarity(X, Y):
    Y = Y.T    # (1, 256) x (256, n) = (1, n)
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))


output_df = pd.read_csv('output_reidentification.csv')
output_df.rename(columns={'ID': 'Reference ID'}, inplace=True)

for (index, row) in output_df.iterrows():

    if index == 0:
        output_df.loc[index, 'ID'] = output_df.loc[index, 'Reference ID']

    else:
        similarity = cos_similarity(output_df.loc[index,'Face_Vectors'], output_df.loc[index+1,'Face_Vectors'])
        if similarity == 1:
            print ("Replaced ID at Index: ", index)
            output_df.loc[index, 'ID'] = output_df.loc[index - 1, 'ID']
            output_df.loc[index, 'Location_History'] += output_df.loc[index - 1, 'Last_Location']
        else:
            output_df.loc[index, 'ID'] = output_df.loc[index, 'Reference ID']

unique_id_df = output_df.drop_duplicates(subset=['ID'], keep='last', inplace=False).reset_index(drop=True)

print(unique_id_df)