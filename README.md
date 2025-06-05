# NYC-Taxi-Duration-Prediction-An-Apache-Airflow-Project

This project builds a machine learning pipeline to predict the duration of taxi rides in New York City using historical data using:

## 🔧 Tech Stack

- Python
- Pandas, Scikit-learn
- MLflow
- Parquet files (data source)
- WSL / Ubuntu environment

- **Apache Airflow**
  
  ![Alt Text](pinwheel.gif) 

## 🔑 MLOps Principles Applied

This project demonstrates key MLOps practices:

- **Reproducibility**: Training scripts accept parameters (`--year`, `--month`) to ensure consistent runs with different data.
- **Experiment Tracking**: Uses MLflow to log parameters, metrics (e.g., RMSE), and artifacts like the trained model and vectorizer.
- **Automation & Orchestration**: A DAG in Apache Airflow automates the full ML pipeline — from data loading to model training and logging.
- **Scalability & Scheduling**: The pipeline can be triggered for different time periods and scheduled for regular execution.

## 📁 Project Structure

- `scripts/train.py`: Train a Linear Regression model and log to MLflow.
- `dags/train_taxi_model_dag.py`: Airflow DAG to schedule training jobs monthly.
- `requirements.txt`: All dependencies.

```
├── airflow/
│   └── dags/
│       └── train_taxi_model_dag.py
├── scripts/
│   └── train.py
├── mlflow_artifact/
│   ├── model.pkl
│   └── dv.pkl
├── README.md
└── requirements.txt
```

## 📊 How It Works

1. **train.py** loads NYC taxi trip data for a given month/year.
2. Preprocesses data and trains a linear regression model.
3. Logs metrics, model, and vectorizer using MLflow.
4. **train_taxi_model_dag.py** defines a DAG that runs the training script in Airflow.


## 🚀 Getting Started


&gt; **Note:** All commands should be run inside your WSL (Windows Subsystem for Linux) terminal or your preferred Linux environment where Airflow and the project dependencies are installed.

### Prerequisites

- Python 3.8+ installed  
- A virtual environment (e.g., `venv` or `conda`) set up with required packages  
- Apache Airflow installed and initialized  
- MLflow tracking server running (if using remote tracking)  

### Order of Execution

1. **Activate your virtual environment and install dependencies:**

   ```bash
   source mlops-w3/bin/activate
   ```

   ```bash
   pip install -r requirements.txt
   ```


1. **Start MLflow server in a dedicated terminal:**

   ```bash
   mlflow server 
     --backend-store-uri sqlite:///mlflow.db 
     --default-artifact-root ./mlflow_artifact 
     --host 0.0.0.0 
     --port 5000
   ```

2. **Initialize Airflow (only once, or if you reset the database):**

   ```bash
   airflow db init
   ```

3. **Start the Airflow scheduler and webserver in separate terminals:**

   Terminal 1:
   ```bash
   airflow scheduler
   ```

   Terminal 2:
   ```bash
   airflow webserver --port 8080
   ```

5. **Open Airflow UI:**

   Open your browser and navigate to http://localhost:8080 to access the Airflow web interface.

6. **Trigger the DAG:**

   You can trigger the DAG manually from the Airflow UI or using the CLI command:

   ```bash
   airflow dags trigger train_taxi_model_dag
   ```

7. **Alternatively, run the training script directly to test:**

   ```bash
   python3 scripts/train.py --year 2023 --month 3
   ```

---

> **Notes:**  
> - MLflow server, Airflow webserver, and scheduler each require their own terminal session.  
> - Make sure to activate your virtual environment (`source mlops-w3/bin/activate`) in each terminal before running commands.  
> - Start MLflow server **before** running the training script to avoid errors connecting to the tracking server.  
> - Follow this sequence every time you restart your environment or machine to ensure Airflow and MLflow run correctly.
