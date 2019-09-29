import sqlalchemy as db
from flask import Flask, request
app = Flask(__name__)

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
        unit_id = request.args['unit_id']  # Required Key
        website = request.args.get('website')
        send_unit_data()


def push_data(request_params):
    query = db.insert(metrics) 
    values_list = [request_params]
    results = connection.execute(query,values_list)
    print("Results of Pushed Data: ", results)


def send_unit_data():
    pass


app.run()