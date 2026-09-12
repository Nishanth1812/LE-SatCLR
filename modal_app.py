import modal


APP_NAME = "le-satclr"
VOLUME_NAME = "le-satclr-data"
VOLUME_PATH = "/vol"
OUTPUT_VOLUME_NAME = "le-satclr-outputs"
OUTPUT_PATH = "/outputs"
RESULTS_PATH = f"{OUTPUT_PATH}/results"
MODELS_PATH = f"{OUTPUT_PATH}/models"
MODEL_NAME_TEMPLATE = "{experiment}_seed42_best.pt"
MLFLOW_SECRET_NAME = "le-satclr-mlflow"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)
mlflow_secret = modal.Secret.from_name(MLFLOW_SECRET_NAME)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.14.0",
    "numpy==2.5.3",
    "mlflow==3.16.0",
    "torchvision==0.29.0",
    "umap-learn==0.5.12",
).add_local_python_source("src")


@app.function(
    image=image,
    gpu="T4",
    timeout=10 * 60,
    secrets=[mlflow_secret],
    volumes={VOLUME_PATH: data_volume, OUTPUT_PATH: output_volume},
)
def smoke_test():
    import os
    import pathlib

    import mlflow
    import torch

    pathlib.Path(VOLUME_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(VOLUME_PATH, "modal_setup_ok.txt").write_text(
        f"torch={torch.__version__}\ncuda={torch.cuda.is_available()}\n",
        encoding="utf-8",
    )
    data_volume.commit()
    output_volume.commit()

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("le-satclr-setup")
    artifact_path = pathlib.Path("/tmp/mlflow_smoke_test.txt")
    artifact_path.write_text("Modal can log artifacts to MLflow.\n", encoding="utf-8")

    with mlflow.start_run(run_name="modal_smoke_test") as run:
        mlflow.log_params({"seed": 42, "stage": "modal_setup"})
        mlflow.log_metric("cuda_available", float(torch.cuda.is_available()))
        mlflow.log_artifact(str(artifact_path))
        print(f"mlflow tracking URI: {tracking_uri}")
        print(f"mlflow run ID: {run.info.run_id}")


@app.local_entrypoint()
def main(stage: str = 'setup', epochs: int = 20, batch_size: int = 128,
         label_percent: int = 1, encoder_checkpoint: str = '', max_batches: int = 0,
         policy: str = 'standard', ssl_scope: str = 'all'):
    if stage == 'setup':
        smoke_test.remote()
    else:
        train_remote.remote(stage,epochs,batch_size,label_percent,encoder_checkpoint,max_batches,policy,ssl_scope)


@app.function(image=image, gpu='T4', timeout=5*60*60, secrets=[mlflow_secret],
              volumes={VOLUME_PATH:data_volume, OUTPUT_PATH:output_volume})
def train_remote(stage,epochs,batch_size,label_percent,encoder_checkpoint,max_batches,policy,ssl_scope):
    from pathlib import Path
    import zipfile
    from src.config import Config
    from src.training import train
    data_root = Path(VOLUME_PATH)/'eurosat'
    if not (data_root/'EuroSAT_RGB').exists():
        with zipfile.ZipFile(data_root/'EuroSAT_RGB.zip') as archive:
            for member in archive.infolist():
                if not (data_root/member.filename).resolve().is_relative_to(data_root.resolve()):
                    raise ValueError('Archive contains an unsafe path')
            archive.extractall(data_root)
        data_volume.commit()
    config = Config(stage=stage,epochs=epochs,batch_size=batch_size,label_percent=label_percent,
                    max_batches=max_batches,policy=policy,ssl_scope=ssl_scope)
    return train(config,data_root,OUTPUT_PATH,encoder_checkpoint or None,commit=output_volume.commit)
