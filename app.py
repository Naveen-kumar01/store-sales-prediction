from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from trainingModel import trainModel
from train_validation_insertion import train_validation
import pickle
import pandas as pd


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('head.html')

def ValuePredictor(to_predict_list):
	loaded_model = pickle.load(open("models/random_forest.pkl", "rb"))
	result = loaded_model.predict(to_predict_list)
	return result[0]

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            train_valObj = train_validation(path)

            train_valObj.train_validation() #calling the training_validation function


            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


@app.route("/predict", methods=['POST'])
@cross_origin()
def getprediction():
    if (request.method == 'POST'):
        pred_list = request.form.to_dict()
        pred_list['Item_Weight'] = float(pred_list['Item_Weight'])
        pred_list['Item_Visibility'] = float(pred_list['Item_Visibility'])
        pred_list['Item_MRP'] = float(pred_list['Item_MRP'])
        pred_list['Outlet_Establishment_Year'] = int(pred_list['Outlet_Establishment_Year'])
        sample_input = pd.DataFrame(pd.Series(pred_list)).T
        result = ValuePredictor(sample_input)
    return render_template("result.html", prediction = result)


if __name__ == '__main__':
    app.run()

port = int(os.getenv("PORT",8000))
if __name__ == "__main__":
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
