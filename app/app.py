from fastapi import FastAPI, HTTPException
import pandas as pd
from back import Back

app = FastAPI()
back = Back()

@app.post("/train-model/")
async def train_model(input_data: dict):
    try:
        raw_data_path = input_data.get("raw_data_path")
        review_conditions = input_data.get("review_conditions")
        if raw_data_path :
            model_id, test_loss, data_snapshot = back.raw_data_processing(raw_data_path,review_conditions)
        return {"model_id": model_id,"test_loss" : test_loss, "data_processed": data_snapshot}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/get-outliers-original-format/")
async def predict_outliers(input_data: dict):
    try:
        model_id = input_data.get("model_id")
        outliers_percentile = input_data.get("outliers_percentile")
        data_processed = input_data.get("data_processed")
        outliers = back.get_outliers(model_id,outliers_percentile,data_processed)
        return {"outliers": outliers}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/get-outliers-drift-score/")
async def report_outliers(input_data: dict):
    try:
        model_id = input_data.get("model_id")
        outliers_percentile = input_data.get("outliers_percentile")
        data_processed = input_data.get("data_processed")
        report = back.report_outliers(model_id,outliers_percentile,data_processed)
        return {"report drift score": report}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))