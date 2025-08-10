from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, pandas as pd

app = FastAPI()

# TODO: ระบุโดเมนจริงของคุณให้แคบที่สุดตอนโปรดักชัน
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ช่วง dev ใช้ * ได้ แต่โปรดจำกัดเมื่อขึ้น prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดลที่ฝึกแล้ว
model = joblib.load("heart_disease_ann_model.pkl")

class Features(BaseModel):
    age:int; sex:int; cp:int; trestbps:int; chol:int; fbs:int
    restecg:int; thalach:int; exang:int; oldpeak:float; slope:int; ca:int; thal:int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(feat: Features):
    df = pd.DataFrame([feat.dict()])
    proba = float(model.predict_proba(df)[:,1][0])
    pred  = int(proba >= 0.5)
    return {"prediction": pred, "risk_score": proba}
