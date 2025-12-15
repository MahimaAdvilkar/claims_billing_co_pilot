import mlflow
from datetime datetime import datetime

mlflow.set_tracking_uri("mlruns")

class RunLogger:
    def __init__(self, run_name: str = None):
        self.run = mlflow.start_run(run_name=run_name or datetime.utcnow().isoformat())

    def log_params(self, **kwargs):
        mlflow.log_params(kwargs)

    def log_metrics(self, **kwargs):
        mlflow.log_metrics(kwargs)

    def end(self):
        mlflow.end_run()
