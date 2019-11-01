import sqlalchemy as db
from flask import Flask, request
app = Flask(__name__)
import json

engine = db.create_engine('mysql://admin:piedpiper@dashboard-db.cabpqqhktlmn.us-east-1.rds.amazonaws.com/HappySpace_Metrics')
connection = engine.connect()
metadata = db.MetaData()

@app.route('/upload', methods=['POST'])
def upload():
    print("Request DATA: ", request.json)
    if request.method == 'POST':
        if request.is_json:
            push_data(request.json)
            return "Data Uploaded to SQL Server!"
        else:
            return "application/json data only"
    return "POST ONLY" 

@app.route('/fetch', methods=['GET'])
def fetch():
    if request.method == 'GET':
        request_params = {}
        view = request.args['view']  
        return send_unit_data(view)


def push_data(request_params):
    metrics = db.Table('LiveData', metadata, autoload=True, autoload_with=engine)

    query = db.insert(metrics) 
    values_list = request_params
    results = connection.execute(query,values_list)
    print("Results of Pushed Data: ", results)


def send_unit_data(view):
    metrics = db.Table(view, metadata, autoload=True, autoload_with=engine)
    query = db.select([metrics])
    results = connection.execute(query)
    return json.dumps([(dict(row.items())) for row in results], default=str)
    
app.run()