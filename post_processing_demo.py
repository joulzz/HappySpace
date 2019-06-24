from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import time

def center_node(x):
    x = ast.literal_eval(x)
    x = (((int(x[0][0])+int(x[1][0]))/2),((int(x[0][1])+int(x[1][1]))/2))
    return x


def timestamp_to_seconds(output_df, index):
    timestamp_str = output_df.loc[index, 'Timestamp']
    actual_time = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    stime = time.strftime("%s", actual_time)
    print("Time: (in seconds)", float(stime))
    return float(stime)


if __name__ == "__main__":
    output_df = pd.read_csv('2018-11-17.csv')
    # Node- (x,y) center of the face bounding box
    output_df['Node'] = pd.DataFrame(output_df['Last_Location'].values.tolist())
    output_df['Node'] = output_df['Node'].apply(center_node)

    # Distance- Euclidean Distance between two nodes
    for (index, row) in output_df.iterrows():
        if index == len(output_df) - 1:
            node_a = np.array(output_df.loc[index,'Node'])
            node_b = np.array(output_df.loc[0,'Node'])

        else:
            node_a = np.array(output_df.loc[index,'Node'])
            node_b = np.array(output_df.loc[index+1,'Node'])
        
        output_df.loc[index,'Distance'] = np.linalg.norm(node_a - node_b)
    print(output_df)

    moving_average = output_df['Distance'].mean()
    print("Moving Average", moving_average)

    total_range = (output_df['Distance'].max() - output_df['Distance'].min(skipna=True))
    print("Range", range)

    variance = output_df['Distance'].var(skipna=True)
    print("Variance", variance)

    # Unique ID Calibration

    R = (total_range * 1/100)
    print("Active Range: ", R)


    output_df.rename(columns={'ID': 'Reference ID'}, inplace=True)

    V = 5 
    print("Turnover Rate ( ms )", V)

    for (index, row) in output_df.iterrows():

        if index == 0:
            output_df.loc[index, 'ID'] = output_df.loc[index, 'Reference ID']

        else:
    
            time_difference = timestamp_to_seconds(output_df, index) - timestamp_to_seconds(output_df, index - 1)
            print("Time Difference: ", time_difference)
            if (output_df.loc[index,'Distance'] < R and time_difference < V):
                print ("Replaced ID at Index: ", index)
                output_df.loc[index,'ID']= output_df.loc[index-1,'ID']
            else:
                output_df.loc[index, 'ID'] = output_df.loc[index,'Reference ID']



    unique_id_df = output_df.drop_duplicates(subset=['ID'], keep='last', inplace=False).reset_index(drop=True)

    print(unique_id_df)


    ## Scores
    for (index, row) in unique_id_df.iterrows():
        if index == 0:
            unique_id_df.loc[index, 'Sociability'] = 0
            unique_id_df.loc[index, 'Relaxed'] = 0
            unique_id_df.loc[index, 'Sociability Score'] = 0
            unique_id_df.loc[index, 'Relaxed Score'] = 0


        else:
            unique_id_df.loc[index, 'Sociability'] = 1 if unique_id_df.loc[index, 'Distance'] <= moving_average else 0
            unique_id_df.loc[index, 'Relaxed'] = 1 if unique_id_df.loc[index, 'Distance'] > moving_average else 0
            unique_id_df.loc[index, 'Sociability Score'] = output_df[output_df["ID"] == index]["Smiles_Detected"].sum() if unique_id_df.loc[index, 'Sociability'] else 0
            unique_id_df.loc[index, 'Relaxed Score'] = output_df[output_df["ID"] == index]["Smiles_Detected"].sum() if unique_id_df.loc[index, 'Relaxed'] else 0
        
    print(unique_id_df)

    print("Dynamic Score: ", len(unique_id_df))
    print("Happiness Score: ",  output_df['Smiles_Detected'].sum())

    # Unique Distributions Plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_coord = []
    y_coord = []


    def axis_coordinates(x):
        x_coord.append(x[0])
        y_coord.append(x[1])


    unique_id_df['Node'].apply(axis_coordinates)

    ax.scatter3D(x_coord, y_coord, unique_id_df['ID'], color='blue', marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('ID')

    plt.show()






