from fastapi import FastAPI, Form, HTTPException
from spam_classifier_service import SpamClassifierService

app = FastAPI()
spam_classifier_service = SpamClassifierService()

@app.post("/train_model")
def train_model(data_path: str = Form("spam.csv")):
    try:
        spam_classifier_service.train_model(data_path)
        score = spam_classifier_service.get_score()
        return {"message": "Model trained successfully","score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/predict_spam")
def predict_spam(message: str = Form(...), retrain: bool = Form(False)):
    try:
        if retrain:
            spam_classifier_service.train_model(data_file_path)

        prediction = spam_classifier_service.predict_spam(message)
        return {"message": message, "category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")