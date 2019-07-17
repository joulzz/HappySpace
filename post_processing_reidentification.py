import pandas as pd
import numpy as np
from ast import literal_eval

def cos_similarity(X, Y):
    X = X.strip("[").strip("]")
    Y = Y.strip("[").strip("]")
    X = np.fromstring(X, sep=' ')
    Y = np.fromstring(Y,sep=' ')
    Y = np.transpose(Y)
    print np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))


output_df = pd.read_csv('output_reidentification.csv')
output_df.rename(columns={'ID': 'Reference ID'}, inplace=True)

for (index, row) in output_df.iterrows():

    if index == 0:
        output_df.loc[index, 'ID'] = output_df.loc[index, 'Reference ID']

    else:
        # output_df['Face_Vectors'] = pd.eval(output_df['Face_Vectors'])

        similarity = cos_similarity(output_df.loc[index,'Face_Vectors'], output_df.loc[index-1,'Face_Vectors'])
        if similarity == 1:
            print ("Replaced ID at Index: ", index)
            output_df.loc[index, 'ID'] = output_df.loc[index - 1, 'ID']
            output_df.loc[index, 'Location_History'] += output_df.loc[index - 1, 'Last_Location']
        else:
            output_df.loc[index, 'ID'] = output_df.loc[index, 'Reference ID']

unique_id_df = output_df.drop_duplicates(subset=['ID'], keep='last', inplace=False).reset_index(drop=True)

print(unique_id_df)