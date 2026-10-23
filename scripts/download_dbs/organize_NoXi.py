# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 05/06/2025
# version = '0.1'
# ---------------------------------------------------------------------------
"""
import os
import sys

file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(file_path)
sys.path.append(os.path.abspath(os.path.join(file_path, '..', '..', '..', '..', 'common')))
sys.path.append(os.path.abspath(os.path.join(file_path, '..', '..', '..', '..', 'configs')))

import shutil
import csv
import json
from pathlib import Path
from natsort import natsorted
from pydub import AudioSegment
import torchaudio

from configuration import config
from src.log.log import getLogger, log_config_file_path

db_name = "NoXi"
config_data = config["database"][db_name]
logger = getLogger(a_instance_name=f"Download_{db_name}_dataset",
                   a_config_file_path=log_config_file_path)

def merge_audios(wav1: Path, wav2: Path, output_path: Path):
    audio1 = AudioSegment.from_wav(wav1)
    audio2 = AudioSegment.from_wav(wav2)
    audio1 = audio1[:min(len(audio1), len(audio2))]
    audio2 = audio2[:min(len(audio1), len(audio2))]
    mix = AudioSegment.from_mono_audiosegments(audio1, audio2)
    mix.export(output_path / "audio_mix.wav", format="wav")


def get_vad_data(src_path):
    vad_list = load_vad_csv(src_path, ' ')

    # return 0, 0, [0,0,0]
    return 0, 0, vad_list


def load_vad_csv(src_path, delimiter=','):
    src_data = load_csv(src_path, delimiter)

    tgt_data = [[], []]
    for row in src_data:
        #if row[-1] == '[speech]':
        if 'novice' in str(src_path.absolute()):
            tgt_data[0].append([float(row[1]), float(row[2])])
        else:
            tgt_data[1].append([float(row[1]), float(row[2])])

    return tgt_data


def load_csv(src_path, delimiter=','):
    with open(src_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        src_data = [x for x in reader]

    return src_data


def copy_files(source_files, dest_path):
    filtered_files = [file for file in source_files if file.endswith(('.wav', '.mp4', ".csv"))]
    for file in filtered_files:
        if os.path.exists(os.path.join(dest_path, os.path.basename(file))):
            logger.LOG_DEBUG(f"File '{file}' already exists in '{dest_path}'. Skipping copy.")
            continue

        shutil.copy(file, dest_path)
        logger.LOG_DEBUG(f"Copy file '{file}' into folder '{dest_path}'")


if __name__ == "__main__":
    db_path = Path(config_data["path_to_data"].rsplit("data", 1)[0])
    db_name_folder = "NoXi+J"
    sessions = "sessions"
    original_path = Path(os.path.join(config_data["path_to_data"].rsplit(db_name_folder, 1)[0], db_name_folder, "MM25_public_noxi"))
    db_path.mkdir(parents=True, exist_ok=True)

    db_features = Path(os.path.join(file_path, "..", "..", "..", "..", "external", "turntaking", "noxi"))
    if not db_features.exists():
        logger.LOG_ERROR(f"Could not find original VAD labels in {db_features}")
        vad_sessions = []
    else:
        vad_sessions = os.listdir(db_features)
        vad_sessions = [x for x in vad_sessions if '.gitkeep' not in x]
        vad_sessions = natsorted(vad_sessions, key=lambda x: (0, x) if x.startswith("Paris_") else (1, x))

    output_json = Path(os.path.join(db_path, "session_correspondence.json"))
    mapping = {}
    modes = ["train", "val", "test-base", "test-additional"]
    for mode in modes:
        original_mode_path = original_path / mode
        mode_path = db_path / sessions
        if not original_mode_path.exists():
            logger.LOG_ERROR(f"Original path {original_mode_path} does not exist.")
            continue

        sessions_subfolders = natsorted(os.listdir(original_mode_path))
        for session in sessions_subfolders:
            session_folder = original_mode_path / session
            logger.LOG_INFO(f"Processing session {session}")
            files_move = [os.path.join(session_folder, x) for x in os.listdir(session_folder)]
            try:
                dest_session_path = mode_path / session
                dest_session_path.mkdir(parents=True, exist_ok=True)
                copy_files(files_move, dest_session_path)
            except Exception as e:
                logger.LOG_ERROR(f"Error copying files from {session_folder} to {mode_path}: {e}")
                continue

            # Find the correspondence between files and VAD labels
            for idx, vad_session in enumerate(vad_sessions):
                vad_session = Path(os.path.abspath(os.path.join(db_features, vad_session)))

                separated = str(vad_session.absolute()).split(os.sep)
                session_name = separated[-1]
                new_session_folder = mode_path / session_name
                mix_wav = vad_session / "audio_mix.wav"

                # Expert
                expert_wav = dest_session_path / "expert.audio.wav"
                x, sr = torchaudio.load(expert_wav.absolute())
                expert_audio_duration = x.shape[1] / sr

                expert_vad = vad_session / "vad_expert.txt"
                start_int, end_int, vad_list = get_vad_data(expert_vad)
                valid_list = vad_list[0]
                if len(valid_list) == 0:
                    valid_list = vad_list[1]

                expert_last_timestamp = round(valid_list[-1][-1], 4)

                # novice
                novice_wav = dest_session_path / "novice.audio.wav"
                x, sr = torchaudio.load(novice_wav.absolute())
                novice_audio_duration = x.shape[1] / sr

                novice_vad = vad_session / "vad_novice.txt"
                start_int, end_int, vad_list = get_vad_data(novice_vad)
                valid_list = vad_list[0]
                if len(valid_list) == 0:
                    valid_list = vad_list[1]

                novice_last_timestamp = round(valid_list[-1][-1], 4)

                result_expert = abs(expert_audio_duration - expert_last_timestamp)
                result_novice = abs(novice_audio_duration - novice_last_timestamp)

                if result_expert < 0.1 and result_novice < 0.1:
                    del vad_sessions[idx]
                    if new_session_folder.exists():
                        mapping[str(dest_session_path)] = {
                            "source": str(dest_session_path),
                            "target": str(vad_session),
                            "audio_mix": str(mix_wav)
                        }
                        try:
                            shutil.rmtree(dest_session_path)
                            logger.LOG_DEBUG(f"Successfully removed {dest_session_path}")
                        except Exception as e:
                            logger.LOG_ERROR(f"Error removing directory: {e}")

                        logger.LOG_INFO(f"Skipping session {session_name} and {vad_session} because it already exists")
                        break

                    logger.LOG_INFO(
                        f"Found session {session_name} {dest_session_path} for audio difference: expert {result_expert} and novice {result_novice}")

                    merge_audios(expert_wav, novice_wav, dest_session_path)

                    for f in ["audio_expert.wav", "audio_novice.wav", "vad_expert.txt", "vad_novice.txt",
                              "non_varbal_expert.csv", "non_varbal_novice.csv"]:
                        src_file = vad_session / f
                        if src_file.exists():
                            tgt_file = dest_session_path / f
                            tgt_file.write_bytes(src_file.read_bytes())

                    mapping[str(dest_session_path)] = {
                        "source": str(dest_session_path),
                        "target": str(vad_session),
                        "audio_mix": str(mix_wav)
                    }
                    try:
                        os.rename(dest_session_path, new_session_folder)
                    except Exception as e:
                        logger.LOG_ERROR(f"Could not rename {dest_session_path} to {new_session_folder}")
                    break


    with output_json.open("w") as f:
        json.dump(mapping, f, indent=2)

    logger.LOG_INFO("All downloads, extractions, and classifications completed.")
