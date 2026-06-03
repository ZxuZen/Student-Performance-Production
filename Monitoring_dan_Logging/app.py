import time
import uvicorn
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

app = FastAPI(title="Student Grade Prediction API")
REQUEST_COUNT = Counter("student_api_requests", "Total request ke API")
PREDICTION_SCORE = Gauge("student_prediction_score", "Hasil prediksi")
PROCESS_TIME = Histogram("student_prediction_process_time", "Waktu yang dibutuhkan untuk prediksi")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
def home():
    REQUEST_COUNT.inc() 
    return {"message": "API Prediksi Nilai Siswa Sedang Berjalan!"}

@app.post("/predict")
def predict(studytime: int, failures: int, absences: int, G1: int, G2: int):
    REQUEST_COUNT.inc() 
    start_time = time.time()
    prediksi_G3 = (G1 + G2) / 2.0 
    PREDICTION_SCORE.set(prediksi_G3)
    PROCESS_TIME.observe(time.time() - start_time)
    
    return {
        "input": {"studytime": studytime, "failures": failures, "absences": absences, "G1": G1, "G2": G2},
        "prediksi_G3": prediksi_G3
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)