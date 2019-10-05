import sqlalchemy as db
from flask import Flask, request
app = Flask(__name__)
import json

engine = db.create_engine('mysql://admin:piedpiper@dashboard-db.cabpqqhktlmn.us-east-1.rds.amazonaws.com/HappySpace_Metrics')
connection = engine.connect()
metadata = db.MetaData()
metrics = db.Table('Metrics', metadata, autoload=True, autoload_with=engine)

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
        unit_id = request.args['unit_id']  # Required Key
        request_params["gender"] = request.form.get('gender')
        request_params["age_group"] = request.form.get('age_group')
        request_params["ethnic_group"] = request.form.get('ethnic_group')
        request_params["sociability"] = request.form.get('sociability')
        request_params["relaxed"] = request.form.get('relaxed')
        return send_unit_data(unit_id, request_params)


def push_data(request_params):
    query = db.insert(metrics) 
    values_list = request_params
    results = connection.execute(query,values_list)
    print("Results of Pushed Data: ", results)


def send_unit_data(unit_id, request_params):
    for param in request_params:
        if param is not None:
            pass
            
    query = db.select([metrics]).where(metrics.columns.unit_id == unit_id)
    results = connection.execute(query)
    return json.dumps([(dict(row.items())) for row in results], default=str)
    
app.run()