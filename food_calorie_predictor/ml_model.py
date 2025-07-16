import requests
import numpy as np
from PIL import Image
import os
import base64
from clarifai.client.model import Model
import asyncio
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

class FoodCaloriePredictor:
    def __init__(self):
        self.pat = 'dbc0023d90c14296801db29753d12b56'
        self.user_id = 'clarifai'
        self.app_id = 'main'
        self.model_id = 'food-item-recognition'
        self.model_version_id = '1d5fd481e0cf4826aa72ec3ff049e044'
        self.food_calories = {
            'apple': 95, 'banana': 105, 'orange': 62, 'grape': 62,
            'strawberry': 4, 'blueberry': 85, 'raspberry': 64,
            'pizza': 266, 'burger': 354, 'hotdog': 151, 'sandwich': 300,
            'pasta': 131, 'rice': 130, 'bread': 79, 'toast': 75,
            'egg': 78, 'chicken': 165, 'beef': 250, 'fish': 206,
            'salad': 20, 'carrot': 25, 'broccoli': 55, 'tomato': 22,
            'potato': 161, 'corn': 88, 'peas': 84, 'beans': 225,
            'milk': 103, 'cheese': 113, 'yogurt': 59, 'ice_cream': 137,
            'cake': 257, 'cookie': 78, 'chocolate': 546, 'candy': 245,
            'coffee': 2, 'tea': 2, 'juice': 111, 'soda': 150,
            'water': 0,
        }

    def predict_food(self, image_path):
        try:
            channel = ClarifaiChannel.get_grpc_channel()
            stub = service_pb2_grpc.V2Stub(channel)
            metadata = (('authorization', 'Key ' + self.pat),)
            userDataObject = resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id)

            with open(image_path, "rb") as f:
                file_bytes = f.read()

            post_model_outputs_response = stub.PostModelOutputs(
                service_pb2.PostModelOutputsRequest(
                    user_app_id=userDataObject,
                    model_id=self.model_id,
                    version_id=self.model_version_id,
                    inputs=[
                        resources_pb2.Input(
                            data=resources_pb2.Data(
                                image=resources_pb2.Image(
                                    base64=file_bytes
                                )
                            )
                        )
                    ]
                ),
                metadata=metadata
            )

            if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                print(post_model_outputs_response.status)
                return None, None, None

            output = post_model_outputs_response.outputs[0]
            concepts = output.data.concepts
            if not concepts:
                return None, None, None
            top_concept = concepts[0]
            food_type = top_concept.name.lower()
            confidence = top_concept.value
            calories = self.food_calories.get(food_type, None)
            return food_type, calories, confidence
        except Exception as e:
            print(f"Error predicting food with Clarifai gRPC: {e}")
            return None, None, None 