# -*- coding: utf-8 -*-
import os


class CNoXiDatabase:
    """
    Database NoXi

    Args:
        a_database_path: dataset path folder
    """

    def __init__(self, a_database_path: str = None):

        self._m_db_name = "Database-NoXi"
        self._m_database_path = a_database_path
        self._m_speakers_path = os.path.join(self._m_database_path, "data", "sessions", "")
        super().__init__()

        self._m_logger.set_new_name(self._m_db_name)
        self._m_logger.LOG_INFO(f"Database {self._m_db_name} definition set successfully")

    def _create_video_paths_list(self) -> list:
        """
        Create video path list

        Returns:
            List with every video file.
        """
        video_paths_list = []
        sessions_folders = sorted([x for x in os.listdir(self._m_speakers_path)])
        for speaker in sessions_folders:
            speaker_path = os.path.join(self._m_speakers_path, speaker)
            if not os.path.exists(speaker_path):
                self._m_logger.LOG_WARNING(
                    f"{speaker} does not exist in {self._m_speakers_path}")
                continue

            speaker_mp4_files = sorted([y for y in os.listdir(speaker_path) if '.mp4' in y])

            for sentence_mp4_file in speaker_mp4_files:
                sentence_video_path = os.path.normpath(
                    os.path.join(speaker_path, sentence_mp4_file))
                video_paths_list.append(sentence_video_path)
        return video_paths_list

    def _create_audio_paths_list(self) -> list:
        """
        Create audio path list

        Returns:
            List with every audio file.
        """
        audio_paths_list = []
        sessions_folders = sorted([x for x in os.listdir(self._m_speakers_path)])
        for speaker in sessions_folders:
            speaker_path = os.path.join(self._m_speakers_path, speaker)
            if not os.path.exists(speaker_path):
                self._m_logger.LOG_WARNING(f"{speaker} does not exist in {self._m_speakers_path}")
                continue

            speaker_wav_files = sorted([y for y in os.listdir(speaker_path) if '.wav' in y])

            for sentence_wav_file in speaker_wav_files:
                sentence_audio_path = os.path.normpath(
                    os.path.join(speaker_path, sentence_wav_file))
                audio_paths_list.append(sentence_audio_path)
        return audio_paths_list

    def _create_audio_labels_from_file(self):
        pass
