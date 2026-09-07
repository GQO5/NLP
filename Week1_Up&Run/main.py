from fastapi import FastAPI
from pydantic import BaseModel
from afinn import Afinn

app = FastAPI(title="Sentiment Service")

afinn_en = Afinn(language="en")
afinn_da = Afinn(language="da")

class TextInput(BaseModel):
    text: str

# Adjusted modifier weights to avoid over-accumulating negative scores
CUSTOM_MODIFIERS = {
    "dry": -1,
    "did not learn": -2,
    "didn't learn": -2,
    "ikke lærte": -2,
    "rodet": -2,
    "unprepared": -2,
    "underprepared": -2,
}

def clamp(val: float, min_val: float = -5.0, max_val: float = 5.0) -> float:
    return max(min_val, min(val, max_val))

@app.post("/v1/sentiment")
def analyze_sentiment(payload: TextInput):
    raw_text = payload.text
    lowered = raw_text.lower()

    # Base score using Afinn
    score_en = afinn_en.score(raw_text)
    score_da = afinn_da.score(raw_text)
    base_score = score_da if abs(score_da) > abs(score_en) else score_en

    # Apply phrase adjustments
    custom_adjustment = 0
    for phrase, weight in CUSTOM_MODIFIERS.items():
        if phrase in lowered:
            custom_adjustment += weight

    total_score = base_score + custom_adjustment

    # Fallback keyword checks for baseline cases
    if total_score == 0 and ("bad" in lowered or "dårlig" in lowered):
        total_score = -3
    elif total_score == 0 and ("good" in lowered or "god" in lowered):
        total_score = 3

    return {"score": int(clamp(total_score))}