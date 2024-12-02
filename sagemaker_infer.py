import os
import json

import torch
from model import BertForWordBoundaryDetection


def model_fn(model_dir):
    """
    Load the model for inference.
    """
    model_path = os.path.join(model_dir, "model/", "pytorch_model.bin")
    model = BertForWordBoundaryDetection()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    if type(request) is not list or any(type(x) is not str for x in request):
        raise TypeError("Request payload must be a list of strings.")

    return request


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    inputs = model.tokenize_function(input_data)
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

    predictions = (torch.sigmoid(outputs) > 0.5).int()

    return predictions


def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response
