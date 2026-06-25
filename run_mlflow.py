import subprocess
import sys
import os

HOST = "127.0.0.1"
PORT = 5000

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    db_uri = f"sqlite:///{project_root}/mlflow.db"
    print(f"Starting MLflow UI at http://{HOST}:{PORT}")
    print(f"Backend: {db_uri}\n")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            db_uri,
            "--host",
            HOST,
            "--port",
            str(PORT),
        ]
    )
