import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import sys

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    csv_path = "student_data_clean.csv"
    try:
        data = pd.read_csv(csv_path)
        print("Data berhasil dimuat!")
    except Exception as e:
        print(f"Error membaca data: {e}")
        sys.exit(1)
    
    X = data.drop(["G3"], axis=1)
    y = data["G3"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predicted = model.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        print("Model berhasil dilatih")