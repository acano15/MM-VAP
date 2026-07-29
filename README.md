![Model pipeline](assets/img/model_pipeline.png)
# README #

Public code artifact for the paper associated with `Multimodal Voice Activity Projection for Turn-Taking in Social Robots with Voice-Activity-Related Pretrained Encoders`.

## Repository Structure

```text
assets/                   Pretrained checkpoints and detector assets
external/                 Required external submodules and third-party code
scripts/                  Dataset preparation and installation helpers
src/                      Training code, configs, models, metrics, and utilities
environment_ubuntu.yml    Ubuntu conda environment
environment_windows.yml   Windows conda environment
requirements_ubuntu.txt   Ubuntu pip requirements
requirements_ubuntu_optional.txt  Optional face-detector backends
requirements_windows.txt  Windows pip requirements
```

# Voice Activity Projection (VAP)

Voice Activity Projection (VAP) is a multimodal system for predicting and analyzing speech activity using audio-visual features.  
It is designed to run on both **Windows** and **Ubuntu** systems with GPU acceleration (CUDA).

## Getting Started

### Supported environment

The tested Ubuntu stack is Ubuntu 22.04, Python 3.10.16, PyTorch 2.6.0, and the
CUDA 12.4 PyTorch wheels. The wheels include the CUDA runtime and cuDNN; a
separate CUDA toolkit installation is not required. GPU execution still needs
an NVIDIA driver compatible with CUDA 12.4. Check it with `nvidia-smi`.

The repository vendors the CPC implementation and obtains the matching
Fairseq source from the `external/av_hubert` submodule. Do not install CPC or a
different Fairseq release separately.

### Installing

#### Windows 10/11

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Verify NVIDIA driver supports CUDA 12.1:
```bash
nvidia-smi
```

To create the environment:
```bash
conda env create -f environment_windows.yml
conda activate mm-vap
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/install_packages.py
pip install --no-build-isolation git+https://github.com/facebookresearch/CPC_audio.git
pip install --no-build-isolation --no-deps git+https://github.com/
pytorch/fairseq.git@afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb
```
#### Ubuntu 22.04

Install the system build, audio, video, Git LFS, and dlib prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake gfortran git git-lfs \
  ffmpeg sox libopenblas-dev liblapack-dev libsndfile1 portaudio19-dev
```

Clone all required source trees and retrieve the checkpoint files stored in
Git LFS:

```bash
git clone --recurse-submodules https://github.com/acano15/MM-VAP.git
cd MM-VAP
git submodule update --init --recursive
git lfs install
git lfs pull
```

If the repository was cloned before Git LFS was installed, confirm that the
files under `assets/` are real binary files rather than small LFS pointer text
files before training.

Create the complete Conda environment from the repository root:

```bash
# Prevent globally sourced ROS packages from leaking into this environment.
unset PYTHONPATH

conda env create -f environment_ubuntu.yml
conda activate mm-vap
```

Alternatively, to create the same Python stack in an existing fresh Python
3.10 environment:

```bash
unset PYTHONPATH
conda create -n mm-vap python=3.10.16 pip=24.0 setuptools=69.5.1 wheel=0.43.0 -y
conda activate mm-vap
python -m pip install -r requirements_ubuntu.txt
```

The default recognition configuration uses OpenCV and dlib and is included in
the main requirements. Install the alternative RetinaFace, FaceAlignment,
MediaPipe, and FaceRecognition backends only if selecting them in
`src/configs/recognition/face_detector.yaml`:

```bash
python -m pip install -r requirements_ubuntu_optional.txt
```

Download the two Whisper-Flamingo checkpoints needed by the default backbone:

```bash
python scripts/download_whisper_flamingo_models.py
```

Validate the installation before preparing data:

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python src/train_mm_vap.py --help
```

The expected version prefix is `2.6.0+cu124`. If CUDA availability is `False`,
the Python environment is valid but the NVIDIA driver or GPU access is not;
training is configured to require a GPU.

Do not install packages one by one after a resolver error. That can hide a
conflict and leave a partially upgraded environment. Delete and recreate the
`mm-vap` environment instead. If ROS paths appear in `python -c "import
sys; print(sys.path)"`, run `unset PYTHONPATH`, deactivate the environment, and
create it again.

---
### ⚙️ Configuration System (Hydra)

Hydra allows you to configure components from YAML files located in your `configs/` directory, such as:

- `recognition/face_detector.yaml`  
- `recognition/landmarks_recognition.yaml`  
- `logger/logger.yaml`  

Each configuration defines the model type and parameters for that module, e.g.:

```yaml
# recognition/face_detector.yaml
face_detector:
  model_name: "RetinaFace"    # or "OpenCV", "DLib", "FaceRecognition"

landmarks_detector:
  model_name: "FaceAlignment" # or "MediaPipe", "DLib"
```

Main configuration files

- `src/configs/config.yaml`: top-level experiment, dataset, and training configuration
- `src/configs/model/model.yaml`: model, backbone, and pretrained asset paths
- `src/configs/events/events.yaml`: event definitions and metrics
- `src/configs/logger/logger.yaml`: logging configuration
- `src/configs/recognition/face_detector.yaml`: face and landmark detector configuration

## Training

Use either the native Ubuntu installation above or the Docker image below. A
Docker build still requires a recursive Git clone with resolved Git LFS assets,
but it installs the system/Python dependencies and downloads the
Whisper-Flamingo checkpoints inside the image.

### Docker

The CUDA 12.4 Ubuntu 22.04 image installs both the default and optional Python
dependencies, includes the Git LFS assets and submodule source, and downloads
the default Whisper-Flamingo checkpoints during the build. The checkpoints are
about 4.5 GB and the tested image is about 11.9 GB, so allow at least 25 GB of
free Docker storage for the build and its temporary layers:

```bash
docker build --progress=plain -f docker/Dockerfile -t mm-vap:cu124 .
```

Run an installation check without a GPU:

```bash
docker run --rm mm-vap:cu124 python -m pip check
```

For training, install NVIDIA Container Toolkit on the host and expose the GPU.
Mount the dataset path used by `src/configs/config.yaml` and increase shared
memory for PyTorch data loading:

```bash
docker run --rm -it --gpus all --shm-size=8g \
  -v /absolute/path/to/datasets:/datasets \
  mm-vap:cu124
```

Inside the container, verify GPU access with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The image's virtual environment is already on `PATH`; skip the Conda activation
step below when working inside the container.

To train the model:

### 1. Activate the environment:
```bash
conda activate mm-vap
```

### 2. Prepare multimodal data (e.g., NoXi dataset):
The active dataset configuration is defined in `src/configs/config.yaml`.

Before running any workflow, update these dataset paths to match the local machine:

- `dataset.linux_root_path`
- `dataset.windows_root_path`
- `dataset.selected_dataset`

Main preparation entrypoints:

```bash
python scripts/prepare_multimodal_noxi.py
```

The preparation scripts generate extracted CSV manifests and face crops under `dataset.extracted_path`.

### 3. Run training:
```bash
python src/train_mm_vap.py
```

The training script:

- loads Hydra configuration from `src/configs/`
- prepares dataset manifests if they do not exist
- trains the selected model
- runs validation
- runs test evaluation

### 4. Evaluation / Inference:

Checkpoint-based evaluation uses the same entrypoint. Set the training config to load a saved checkpoint:

```bash
python src/train_mm_vap.py train.training_features.pretrained=true train.training_features.model_checkpoints_path=/path/to/checkpoint.ckpt
```

## Help

For dependency diagnostics, include the output of `python -m pip check`,
`python --version`, `python -m pip --version`, and `nvidia-smi` in the issue.
Do not regenerate `requirements_ubuntu.txt` with a machine-wide `pip freeze`;
that can capture ROS packages and unrelated system software.

## Authors

- **Antonio Cano Montes** - *PhD Student* -
    [github-profile](https://github.com/acano15)

## Version History

* 0.1
    * Initial Release

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Citation
The paper corresponding to this repository is currently available as a pre-print:
https://arxiv.org/abs/2607.07294

If you use this repository before the final manuscript is public, please contact the authors for the appropriate citation information.

## Contact

For questions about the artifact, reproducibility, or the paper, contact:

- `aantcan@alu.upo.es` | `aantcan@upo.es` | `a.cano@4i.ai`
- `contact@antoniocanomontes.com`
