from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from starlette.responses import JSONResponse, JSONDecodeError
from typing import Any, List, Dict
from dotenv import load_dotenv
import tensorflow as tf
import pandas as pd
import numpy as np
import os


class CorsAndRefererMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, allowed_origins: List[str]):
        super().__init__(app)
        self.allowed_origins = allowed_origins
        
    async def dispatch(self, request, call_next):
        # Check the request origin
        origin = request.headers.get("Origin")
        if origin not in self.allowed_origins:
            raise HTTPException(status_code=403, detail="Origin not allowed")

        # Check the Referer header
        referer = request.headers.get("Referer")
        if not referer:
            raise HTTPException(status_code=403, detail="Referer header missing")
        if referer not in self.allowed_origins:
            raise HTTPException(status_code=403, detail="Where are you from bruh🤨")

        # Call the next middleware in the stack
        response = await call_next(request)

        return response
    
def init_model():
    le = LabelEncoder()
    df = pd.read_csv('clean_data.csv')
    le.fit(df['grade'])
    model = tf.keras.models.load_model("nutrient.h5")
    
    return le, model

class BaseAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_api()
        self.setup_middleware()
        self.le, self.model = init_model()  # Initialize model and LabelEncoder
    
    def setup_api(self):
        self.app.post("/api/v1/nutrient")(self.process_nutrient)
    
    async def process_nutrient(self, request: Request):
        try:
            body = await request.json()
            known_keys = {'fat', 'sugar', 'sodium'}
            val_input = [body[k] for k in body if k in known_keys]
            if len(val_input) != 3:
                raise HTTPException(status_code=400, detail="Please provide values for 'fat', 'sugar', and 'sodium' keys")
            val_input_arr = np.array(val_input, dtype=np.float32).reshape(1, -1)
        
            predictions = self.model.predict(val_input_arr)
            predicted_class = self.le.inverse_transform(np.argmax(predictions, axis=1))[0]
        
            return JSONResponse(content={"result": predicted_class})
        except JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON data provided")
        except KeyError:
            raise HTTPException(status_code=400, detail="Invalid keys provided in JSON data")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def setup_middleware(self):
        load_dotenv()
        allowed_origins = os.getenv("allowed_origins", "").split(",")
        
        self.app.add_middleware(
            CorsAndRefererMiddleware,
            allowed_origins=allowed_origins
        )
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["POST"],
            allow_headers=["referer"]
        )
        

base_api = BaseAPI()
