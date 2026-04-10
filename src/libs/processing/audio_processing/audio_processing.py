# -*- coding: utf-8 -*-
import numpy as np
import librosa


class CAudioProcessing:
    """
    Audio processing
    """
    def __init__(self):
        pass

    @staticmethod
    def resample_audio_signal(audio_signal: np.array(float), sr: int, new_sr: int) -> tuple:
        """
        Librosa audio resample

        Args:
            audio_signal: np.array(float):
            sr: int:
            new_sr: int:

        Returns:
            Tuple with the new signal and the new sample rate.
        """
        y_resample = librosa.resample(audio_signal, orig_sr=sr, target_sr=new_sr,
                                      res_type='kaiser_best')
        return y_resample

    @staticmethod
    def int2float(sound: np.array(float)) -> np.array(float):
        """
        Numpy int to float array

        Args:
            sound: Any:

        Returns:
            The return value. True for success, False otherwise.
        """
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound
