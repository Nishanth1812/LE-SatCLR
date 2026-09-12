import os
from pathlib import Path

import modal


APP_NAME = os.environ.get("LE_SATCLR_APP_NAME", "le-satclr")
VOLUME_NAME = "le-satclr-data"
VOLUME_PATH = "/vol"
OUTPUT_VOLUME_NAME = "le-satclr-outputs"
OUTPUT_PATH = "/outputs"
RESULTS_PATH = f"{OUTPUT_PATH}/results"
MODELS_PATH = f"{OUTPUT_PATH}/models"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.14.0",
    "numpy==2.5.3",
    "mlflow==3.16.0",
    "torchvision==0.29.0",
    "umap-learn==0.5.12",
    "fastapi>=0.141.1",
    "uvicorn>=0.52.4",
    "scikit-learn>=1.9.1",
    "pillow>=12.3.0",
).add_local_python_source("src")

# Bundle the built frontend when present; the dashboard serves API-only otherwise.
if Path("web/dist/index.html").is_file():
    image = image.add_local_dir("web/dist", remote_path="/web/dist")
    WEB_DIST = "/web/dist"
else:
    WEB_DIST = None


@app.function(
    image=image,
    gpu="A10G",
    timeout=10 * 60,
    volumes={VOLUME_PATH: data_volume, OUTPUT_PATH: output_volume},
)
def smoke_test():
    import pathlib

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


@app.local_entrypoint()
def main(stage: str = 'setup', epochs: int = 0, batch_size: int = 0,
         label_percent: int = 1, encoder_checkpoint: str = '', max_batches: int = 0,
         policy: str = 'standard', ssl_scope: str = 'all', isolated_tracking: bool = False):
    if stage == 'setup':
        smoke_test.remote()
    else:
        train_remote.remote(stage,epochs,batch_size,label_percent,encoder_checkpoint,max_batches,policy,ssl_scope,isolated_tracking)


@app.function(image=image, gpu='A10G', timeout=5*60*60,
              volumes={VOLUME_PATH:data_volume, OUTPUT_PATH:output_volume})
def downstream_remote(epochs: int, batch_size: int, label_percent: int, encoder_checkpoint: str,
                      max_batches: int, policy: str, ssl_scope: str, isolated_tracking: bool = False):
    """Run probe, finetune and baseline for one label budget from the same SSL checkpoint."""
    from src.config import resolve_batch_size, resolve_epochs
    epochs, batch_size = int(epochs), int(batch_size)
    label_percent, max_batches = int(label_percent), int(max_batches)
    if label_percent not in (1,10):
        raise ValueError('Label budget must be 1 or 10')
    if not encoder_checkpoint:
        raise ValueError('An SSL checkpoint is required')
    results = []
    for stage in ('probe','finetune','baseline'):
        checkpoint = encoder_checkpoint if stage != 'baseline' else ''
        stage_epochs = epochs if epochs not in (None,0) else resolve_epochs(stage,None)
        stage_batch = batch_size if batch_size not in (None,0) else resolve_batch_size(stage,None)
        results.append(train_remote.local(stage,stage_epochs,stage_batch,label_percent,checkpoint,
                                          max_batches,policy,ssl_scope,isolated_tracking))
    return results


@app.function(image=image, gpu='A10G', timeout=5*60*60,
              volumes={VOLUME_PATH:data_volume, OUTPUT_PATH:output_volume})
def train_remote(stage: str, epochs: int, batch_size: int, label_percent: int, encoder_checkpoint: str,
                 max_batches: int, policy: str, ssl_scope: str, isolated_tracking: bool = False):
    from pathlib import Path
    import zipfile
    from src.config import Config, resolve_batch_size, resolve_epochs
    from src.training import train
    epochs = resolve_epochs(stage, int(epochs) if isinstance(epochs, str) else epochs)
    batch_size = resolve_batch_size(stage, int(batch_size) if isinstance(batch_size, str) else batch_size)
    label_percent = int(label_percent) if isinstance(label_percent, str) else label_percent
    max_batches = int(max_batches) if isinstance(max_batches, str) else max_batches
    if isolated_tracking and not max_batches:
        raise ValueError('Isolated tracking is only allowed for smoke tests')
    data_root = Path(VOLUME_PATH)/'eurosat'
    if not (data_root/'EuroSAT_RGB').exists():
        with zipfile.ZipFile(data_root/'EuroSAT_RGB.zip') as archive:
            for member in archive.infolist():
                if not (data_root/member.filename).resolve().is_relative_to(data_root.resolve()):
                    raise ValueError('Archive contains an unsafe path')
            archive.extractall(data_root)
        data_volume.commit()
    config = Config(stage=stage,epochs=epochs,batch_size=batch_size,label_percent=label_percent,
                    max_batches=max_batches,policy=policy,ssl_scope=ssl_scope,workers=4)
    # An explicit offline diagnostic, never a silent fallback for real runs.
    tracking_uri = 'sqlite:////tmp/le-satclr-smoke.db' if isolated_tracking else None
    return train(config,data_root,OUTPUT_PATH,encoder_checkpoint or None,tracking_uri,commit=output_volume.commit)


@app.function(image=image, timeout=30*60, min_containers=1,
              volumes={VOLUME_PATH:data_volume, OUTPUT_PATH:output_volume})
@modal.asgi_app()
def dashboard_app():
    """Serve the evaluation dashboard (API + bundled frontend) on Modal."""
    from src.dashboard import DashboardService, create_app
    service = DashboardService(
        root='/',
        data_root=f'{VOLUME_PATH}/eurosat/EuroSAT_RGB',
        split_path=f'{OUTPUT_PATH}/results/splits_seed42.json',
        model_dirs=[f'{OUTPUT_PATH}/models'],
    )
    return create_app(service, WEB_DIST)
