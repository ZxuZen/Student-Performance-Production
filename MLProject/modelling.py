import os
import sys
import mlflow
import argparse 
import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_url", type=str, default="student_data_clean.csv")
    parser.add_argument("target_var", type=str, default="G3")
    args = parser.parse_args()
    
    try:
        data = pd.read_csv(args.csv_url)
        print(f"Berhasil memuat data {args.csv_url}")
    except Exception as e:
        print(f"Error membaca data: {e}")
        sys.exit(1)
    
    X = data.drop([args.target_var], axis=1)
    y = data[args.target_var]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predicted = model.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")

        with open("../run_id.txt", "w") as f:
            f.write(run.info.run_id)