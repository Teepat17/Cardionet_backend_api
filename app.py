# app.py
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Literal

HERE = Path(__file__).resolve().parent
DETAILED_MODEL_PATH = Path(os.getenv("DETAILED_MODEL_PATH", HERE / "heart_detailed_pipeline.pkl"))
COARSE_MODEL_PATH   = Path(os.getenv("COARSE_MODEL_PATH",   HERE / "heart_coarse_pipeline.pkl"))

if not DETAILED_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing detailed model at {DETAILED_MODEL_PATH}")
if not COARSE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing coarse model at {COARSE_MODEL_PATH}")

pipe_detailed = joblib.load(DETAILED_MODEL_PATH)
pipe_coarse   = joblib.load(COARSE_MODEL_PATH)

DETAILED_FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
COARSE_FEATURES = [
    "age_bin","trestbps_bin","chol_bin","thalach_bin","oldpeak_bin",
    "fbs_cat","exang_cat","sex_cat",
    "cp_cat","restecg_cat","slope_cat","ca_bin","thal_cat"
]
CHOICES = {
    "age_bin":      ["<40","40-49","50-59","60-69","≥70"],
    "trestbps_bin": ["<120","120-139","140-159","≥160"],
    "chol_bin":     ["<200","200-239","≥240"],
    "thalach_bin":  ["<120","120-149","150-179","≥180"],
    "oldpeak_bin":  ["0","0.1-1.9","2.0-3.9","≥4.0"],
    "fbs_cat":      ["ปกติ","สูง"],
    "exang_cat":    ["ไม่เจ็บหน้าอกตอนออกแรง","เจ็บหน้าอกตอนออกแรง"],
    "sex_cat":      ["หญิง","ชาย"],
    "cp_cat":       ["เจ็บหน้าอกแบบทั่วไป","เจ็บหน้าอกแบบไม่คงที่","ปวดแน่น/ไม่ชัดเจน","ไม่เจ็บหน้าอก"],
    "restecg_cat":  ["คลื่นหัวใจปกติ","ผิดปกติเล็กน้อย","ผิดปกติชัดเจน"],
    "slope_cat":    ["ลาดลง","แบน","ลาดขึ้น"],
    "ca_bin":       ["ไม่มีเส้นเลือดตีบ","มีอย่างน้อย 1 เส้น"],
    "thal_cat":     ["รอยโรคคงที่","ปกติ","รอยโรคกลับได้"],
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS","*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utils ----------
def _bin_age(age: float) -> str:
    if age < 40: return "<40"
    if age <= 49: return "40-49"
    if age <= 59: return "50-59"
    if age <= 69: return "60-69"
    return "≥70"

def _is_coarse(payload: Dict[str, Any]) -> bool:
    return any(k in payload and payload[k] is not None for k in COARSE_FEATURES if k != "age_bin")

def _validate_coarse_values(row: Dict[str, Any]):
    for k, allowed in CHOICES.items():
        if k in row and row[k] is not None and row[k] not in allowed:
            raise HTTPException(status_code=400, detail=f"{k} must be one of {allowed}, got '{row[k]}'")

# ---------- Schemas ----------
class PredictPayload(BaseModel):
    # เลือกโหมดเองได้: "detailed" | "coarse" | ไม่ส่ง (auto)
    mode: Optional[Literal["detailed","coarse"]] = Field(default=None)

    # detailed fields (ตัวเลข)
    age: Optional[float] = None
    sex: Optional[float] = None
    cp: Optional[float] = None
    trestbps: Optional[float] = None
    chol: Optional[float] = None
    fbs: Optional[float] = None
    restecg: Optional[float] = None
    thalach: Optional[float] = None
    exang: Optional[float] = None
    oldpeak: Optional[float] = None
    slope: Optional[float] = None
    ca: Optional[float] = None
    thal: Optional[float] = None

    # coarse fields (ตัวเลือก)
    age_bin: Optional[str] = None
    trestbps_bin: Optional[str] = None
    chol_bin: Optional[str] = None
    thalach_bin: Optional[str] = None
    oldpeak_bin: Optional[str] = None
    fbs_cat: Optional[str] = None
    exang_cat: Optional[str] = None
    sex_cat: Optional[str] = None
    cp_cat: Optional[str] = None
    restecg_cat: Optional[str] = None
    slope_cat: Optional[str] = None
    ca_bin: Optional[str] = None
    thal_cat: Optional[str] = None

    @validator("*", pre=True, always=True)
    def strip_if_str(cls, v):
        return v.strip() if isinstance(v, str) else v

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(body: PredictPayload):
    data = body.dict(exclude_none=True)

    # 1) ตีความ mode
    mode = data.pop("mode", None)  # เอาออกจาก data เพื่อไม่ให้ไปชนกับฟีเจอร์
    if mode is None:
        # auto: ถ้ามีพวก *_bin / *_cat ให้ถือเป็น coarse, ไม่งั้น detailed
        mode = "coarse" if _is_coarse(data) else "detailed"

    # 2) route ตามโหมด
    if mode == "coarse":
        # สร้าง/ตรวจ age_bin
        if "age_bin" not in data:
            if "age" not in data:
                raise HTTPException(status_code=400, detail="Coarse mode requires 'age' (number) or 'age_bin'.")
            data["age_bin"] = _bin_age(float(data["age"]))
        # ตรวจข้อความตัวเลือก
        _validate_coarse_values(data)
        # เช็ค field ครบ
        missing = [k for k in COARSE_FEATURES if k not in data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields for coarse mode: {missing}")
        row = pd.DataFrame([[data[k] for k in COARSE_FEATURES]], columns=COARSE_FEATURES)
        proba = float(pipe_coarse.predict_proba(row)[:,1][0])
        pred  = int(proba >= 0.5)
        return {"mode":"coarse","prediction":pred,"risk_score":proba}

    elif mode == "detailed":
        missing = [k for k in DETAILED_FEATURES if k not in data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields for detailed mode: {missing}")
        row = pd.DataFrame([[data[k] for k in DETAILED_FEATURES]], columns=DETAILED_FEATURES)
        proba = float(pipe_detailed.predict_proba(row)[:,1][0])
        pred  = int(proba >= 0.5)
        return {"mode":"detailed","prediction":pred,"risk_score":proba}

    else:
        raise HTTPException(status_code=400, detail="mode must be 'detailed' or 'coarse'")
