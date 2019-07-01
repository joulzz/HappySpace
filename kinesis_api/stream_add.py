import boto3
client = boto3.client('kinesis')
import base64
import pandas as pd



def get_partition_key():
    import socket    
    hostname = socket.gethostname()    
    IPAddr = socket.gethostbyname(hostname)  
    return str(IPAddr)


def kinesis_put_data(data):
    response = client.put_record(
        StreamName='HappySpaceStream',
        Data=data,
        PartitionKey=get_partition_key()
    )
    print("****** Kinesis Data Pushed with response : {} ******".format(response))



if __name__ == "__main__":
    data_csv = pd.read_csv("/home/suraj/Downloads/2018-11-19.csv")
    for i in range(len(data_csv.index)):
        row_entry = ("|").join([str(val) for val in data_csv.iloc[i][:].values.tolist()])
        kinesis_put_data(row_entry + "\n")