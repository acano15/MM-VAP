![Model pipeline](assets/img/model_pipeline.png)
# README #

Public code artifact for the paper associated with `Multimodal Voice Activity Projection for Turn-Taking in Social Robots with Voice-Activity-Related Pretrained Encoders`.

## Repository Structure

```text
asset/                    Pretrained checkpoints and detector assets
external/                 Required external submodules and third-party code
scripts/                  Dataset preparation and installation helpers
src/                      Training code, configs, models, metrics, and utilities
environment_ubuntu.yml    Ubuntu conda environment
environment_windows.yml   Windows conda environment
requirements_ubuntu.txt   Ubuntu pip requirements
requirements_windows.txt  Windows pip requirements
```

# Voice Activity Projection (VAP)

Voice Activity Projection (VAP) is a multimodal system for predicting and analyzing speech activity using audio-visual features.  
It is designed to run on both **Windows** and **Ubuntu** systems with GPU acceleration (CUDA).

## Getting Started

### Dependencies
We provide ready-to-use `environment.yml` files for each platform.  
These specify Python version, Conda dependencies, and pip dependencies.

* Windows 10/11 
    * Python 3.10.14
    * Conda 4.14.0
    * CUDA 12.1 (driver ≥ 531.14)
    * cuDNN (via pip/conda with PyTorch)
    * Torch 2.5.1 (GPU build with cu121)
    * Dlib installed via Conda
    * Required pip packages
* Ubuntu 22.04
    * Python 3.10.14
    * Conda (Anaconda/Miniconda)
    * CUDA 12.1 (driver ≥ 525.xx)
    * cuDNN 9.x (installed via pip packages: `nvidia-cudnn-cu12==9.1.0.70`)
    * Torch 2.6.0 (nightly build with cu121)
    * Dlib built from source (GPU support)
    * Additional dependencies: `sox`, `portaudio19-dev`, `libsndfile1`

### Environment Creation

It is recommended to use **conda** for reproducibility and binary dependency management.
```bash
# Create conda environment
conda create -n mm-vap python=3.10.16
conda activate mm-vap
```

### Submodules
Clone the repository and initialize submodules After cloning the repository, make sure to initialize and update the submodules by running:
```bash
git submodule update --init --recursive
```

Required submodules:

- `external/TalkNet-ASD`
- `external/whisper-flamingo`
- `external/av_hubert`

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

Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake gfortran libatlas-base-dev libsndfile1 portaudio19-dev sox
```
Create the environment:
```bash
conda create -n mm-vap python=3.10.16
conda activate mm-vap
pip install -r requirements_ubuntu.txt
pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 torchaudio==2.6.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/install_packages.py
pip install --no-build-isolation git+https://github.com/facebookresearch/CPC_audio.git
pip install --no-build-isolation --no-deps git+https://github.com/pytorch/fairseq.git@afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb
```


#### Fallback: Install One by One

Sometimes bulk installation fails due to version conflicts or OS-specific issues.
In that case, you can install each package individually using the helper scripts:

- Linux / macOS

  Run the Bash script:
  ```bash
  chmod +x install_requirements.sh
  ./install_requirements.sh
  ```
  This script reads requirements_ubuntu.txt and installs each dependency one at a time.

- Windows (PowerShell)

  Run the PowerShell script:
  ```bash
  .\install_requirements.ps1
  ```
  This script reads requirements_windows.txt and installs dependencies one by one.

  If execution policy blocks .ps1 scripts, run:
  ```bash
  powershell -ExecutionPolicy Bypass -File .\install_requirements.ps1
  ```
#### Manual Installation (last resort)

If even the scripts fail, open the requirements_ubuntu.txt or requirements_windows.txt files and run:

pip install <package-name> , for each dependency manually.

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

Make sure to initialize **submodules** and download files managed with **Git LFS** (pretrained models, datasets pointers, etc.):

```bash
git clone https://github.com/acano15/MM-VAP.git
cd MM-VAP

# initialize submodules
git submodule update --init --recursive

# install git lfs and pull large files
git lfs install
git lfs pull
```

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

## Environment Export (optional)

To freeze and share environment:

Create dependencies using `pip list --format=freeze > requirements.txt`

Create environment using `conda env export > environment.yml`

## Help

Any advise for common problems or issues.
```python
command to run if program contains helper info
```

## Authors

- **Antonio Cano Montes** - *PhD Student* -
    [github-profile](https://github.com/acano15)

## Version History

* 0.1
    * Initial Release

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Citation
The paper corresponding to this repository is currently under review and will be made available soon. If you use this repository before the final manuscript is public, please contact the authors for the appropriate citation information.

## Contact

For questions about the artifact, reproducibility, or the paper, contact:

- `aantcan@alu.upo.es`
- `contact@antoniocanomontes.com`
