# app.py
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, Literal

# ---------------- Paths & load models ----------------
HERE = Path(__file__).resolve().parent
DETAILED_MODEL_PATH = Path(os.getenv("DETAILED_MODEL_PATH", HERE / "hgb_monotone_full.pkl"))
COARSE_MODEL_PATH   = Path(os.getenv("COARSE_MODEL_PATH",   HERE / "hgb_monotone_coarse.pkl"))

if not DETAILED_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing detailed model at {DETAILED_MODEL_PATH}")
if not COARSE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing coarse model at {COARSE_MODEL_PATH}")

pipe_detailed = joblib.load(DETAILED_MODEL_PATH)
pipe_coarse   = joblib.load(COARSE_MODEL_PATH)

# ---------------- Feature schemas ----------------
# detailed: ใช้ฟีเจอร์ครบ (ตัวเลขดิบ)
DETAILED_FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

# coarse: ใช้ตัวเลขดิบ แต่ "ตัด" slope, oldpeak, thal ออกให้ตรงกับที่เทรนในโน้ตบุ๊ก
COARSE_FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","ca"
]

# ---------------- FastAPI app ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS","*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Utils ----------------
def _count_present(keys, payload: Dict[str, Any]) -> int:
    return sum(k in payload and payload[k] is not None for k in keys)

# ---------------- Pydantic schema (v1/v2 compatible) ----------------
# รองรับทั้ง Pydantic v1 (@validator) และ v2 (@field_validator)
try:
    from pydantic import BaseModel, Field
    from pydantic import field_validator as _validator  # pydantic v2
    _V2 = True
except Exception:
    from pydantic import BaseModel, Field, validator as _validator  # pydantic v1
    _V2 = False

class PredictPayload(BaseModel):
    # เลือกโหมดเองได้: "detailed" | "coarse" | ไม่ส่ง (auto)
    mode: Optional[Literal["detailed","coarse"]] = None

    # detailed/coarse fields (ตัวเลขดิบทั้งหมด)
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

    if _V2:
        @_validator('*', mode='before')
        @classmethod
        def _strip_if_str(cls, v):
            return v.strip() if isinstance(v, str) else v
    else:
        @_validator('*', pre=True, always=True)
        def _strip_if_str(cls, v):
            return v.strip() if isinstance(v, str) else v

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(body: PredictPayload):
    data = body.dict(exclude_none=True)
    mode = data.pop("mode", None)

    # auto select: นับจำนวนฟิลด์ที่ส่งมาของแต่ละโหมด
    if mode is None:
        detailed_count = _count_present(DETAILED_FEATURES, data)
        coarse_count   = _count_present(COARSE_FEATURES, data)
        # ถ้ากรอกฟีเจอร์ของ coarse มากกว่าหรือเท่ากับ detailed ให้ไป coarse
        mode = "coarse" if coarse_count >= detailed_count else "detailed"

    if mode == "coarse":
        # สำหรับ coarse: ยอมให้บางคอลัมน์เป็น None ได้ (มี imputer ใน pipeline จัดการ)
        row_dict = {k: data.get(k, None) for k in COARSE_FEATURES}
        row = pd.DataFrame([row_dict], columns=COARSE_FEATURES)
        try:
            proba = float(pipe_coarse.predict_proba(row)[:, 1][0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Coarse predict failed: {e}")
        pred  = int(proba >= 0.5)
        return {"mode": "coarse", "prediction": pred, "risk_score": proba, "used_features": COARSE_FEATURES}

    elif mode == "detailed":
        missing = [k for k in DETAILED_FEATURES if k not in data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields for detailed mode: {missing}")
        row = pd.DataFrame([[data[k] for k in DETAILED_FEATURES]], columns=DETAILED_FEATURES)
        try:
            proba = float(pipe_detailed.predict_proba(row)[:, 1][0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Detailed predict failed: {e}")
        pred  = int(proba >= 0.5)
        return {"mode": "detailed", "prediction": pred, "risk_score": proba, "used_features": DETAILED_FEATURES}

    else:
        raise HTTPException(status_code=400, detail="mode must be 'detailed' or 'coarse'")
