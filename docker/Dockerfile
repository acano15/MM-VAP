# syntax=docker/dockerfile:1.7

# PyTorch supplies its own CUDA 12.4 runtime libraries.  This NVIDIA base also
# makes the expected GPU/container contract explicit while avoiding the much
# larger CUDA compiler image; dlib is therefore built as a portable CPU backend.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_PARALLEL_LEVEL=4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_INPUT=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/mm-vap-venv \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=""

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        gfortran \
        git \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        liblapack-dev \
        libopenblas-dev \
        libsndfile1 \
        portaudio19-dev \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        pip==24.0 \
        setuptools==69.5.1 \
        wheel==0.43.0

WORKDIR /workspace/mm-vap

# Install dependencies before copying the source so ordinary code changes can
# reuse this expensive layer. The optional file includes the core file.
COPY requirements_ubuntu.txt requirements_ubuntu_optional.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements_ubuntu_optional.txt \
    && python -m pip check

COPY . .

# Fail at build time when a non-recursive clone or unresolved Git LFS checkout
# was supplied as the Docker context.
RUN test -f external/TalkNet-ASD/talkNet.py \
    && test -f external/av_hubert/avhubert/hubert_asr.py \
    && test -f external/whisper-flamingo/whisper/__init__.py \
    && test "$(stat -c %s assets/cpc/60k_epoch4-d0f474de.pt)" -gt 1000 \
    && test "$(stat -c %s assets/asd/pretrain_TalkSet.model)" -gt 1000 \
    && test "$(stat -c %s assets/vap/vap_state_dict_eng_20hz_2500msec.pt)" -gt 1000

# These checkpoints are required by the default Whisper-Flamingo backbone and
# are deliberately embedded so a created container does not download at run time.
RUN python scripts/download_whisper_flamingo_models.py \
    && test "$(stat -c %s pretrained_models/whisper_flamingo/whisper-flamingo_en-x_small.pt)" -gt 1000 \
    && test "$(stat -c %s pretrained_models/whisper_flamingo/large_noise_pt_noise_ft_433h_only_weights.pt)" -gt 1000

# Build-time smoke tests do not require a GPU. CUDA access is checked after the
# image is launched with `--gpus all`.
RUN python -m pip check \
    && python -c "import torch, torchvision, torchaudio, cv2, dlib, pytorch_lightning; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)" \
    && python -c "from src.libs.model.env.whisper_setup import setup_whisper_flamingo_environment; setup_whisper_flamingo_environment(); import fairseq, hubert_asr, whisper_flamingo_whisper; print('bundled model sources: OK')" \
    && python src/train_mm_vap.py --help >/tmp/mm-vap-help.txt

CMD ["/bin/bash"]
