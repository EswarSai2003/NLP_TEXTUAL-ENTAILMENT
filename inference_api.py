#importing modeules essential for running fast api
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

#importing function whcih we defined in 'predicting_function' file to predict gold label
from predicting_function import predict

#  Class which describes inputs to be taken(premise and hypothesis):
class sentences(BaseModel):
    premise: str 
    hypothesis:str

base_path = '/Users/vkadava/Desktop/PROJECT3/'
model_path = base_path + 'saved_model.pth'

app = FastAPI()

#  Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted gold label
@app.post('/predict')
def predict_relation(data:sentences):
    data = data.dict()
    premise=data['premise']
    hypothesis=data['hypothesis']
    prediction = predict(model_path,premise,hypothesis)
    return {
        'prediction': prediction
    }

#run this code in terminal after running this file to get url link from which 
#we can get predictions by hitting end point 'predict':"uvicorn inference_api:app --reload"



