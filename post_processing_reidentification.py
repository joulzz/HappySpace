import pandas as pd
import numpy as np
import itertools

def cos_similarity(X, Y):

    """ Description

    Functions maps the cosine similarity among the face feature vectors retrieved from detected faces

    :return: Returns the similarity mapped using the cosine of the angle between two vectors projected
    in a multi-dimensional space
    """
    
    X = X.strip("[").strip("]")
    Y = Y.strip("[").strip("]")
    X = np.fromstring(X, sep=' ')
    Y = np.fromstring(Y,sep=' ')
    Y = np.transpose(Y)
    print(np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0)))
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

csv_file_path = 'output_live_test.csv'
output_df = pd.read_csv(csv_file_path)
output_df.rename(columns={'ID': 'Reference ID'}, inplace=True)
indices = output_df.index.values.tolist()
deleted_row_index=[]

for output_df_index in indices:
    if output_df.loc[output_df_index,'Face_Vectors'] == '[]':
        indices.remove(output_df_index)
        output_df.drop(output_df_index, inplace=True)
print(indices)

output_df['ID']= output_df['Reference ID']

for target,feature in itertools.combinations(indices, 2):
    print(target," ",feature)
    similarity = cos_similarity(output_df.loc[target,'Face_Vectors'], output_df.loc[feature,'Face_Vectors'])
    if similarity > 0.7:
        print ("Replaced ID at Index: ", feature, "Reference_ID:",output_df.loc[feature,'Reference ID'])
        output_df.loc[feature, 'ID'] = output_df.loc[target, 'ID']
        output_df.loc[feature, 'Location_History'] += output_df.loc[target, 'Last_Location']
    else:
        output_df.loc[feature, 'ID'] = output_df.loc[feature, 'ID']

print(output_df)
print('Removing Duplicates')
unique_id_df = output_df.drop_duplicates(subset=['ID'], keep='last', inplace=False).reset_index(drop=True)
print(unique_id_df)
unique_id_df.to_csv(csv_file_path)