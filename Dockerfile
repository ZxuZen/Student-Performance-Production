FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install pandas scikit-learn mlflow
CMD ["python", "MLProject/modelling.py", "MLProject/student_data_clean.csv", "G3"]