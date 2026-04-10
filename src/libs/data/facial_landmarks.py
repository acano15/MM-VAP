# -*- coding: utf-8 -*-
from dataclasses import dataclass

# DEFINITION
LIPS_INDEXES = list(range(48, 68))
NOSECHIN_INDEXES = list(range(2, 15)) + list(range(31, 36))


@dataclass(frozen=True)
class EFacialLandmarks:
    JAWLINE_0: int = 0
    JAWLINE_1: int = 1
    JAWLINE_2: int = 2
    JAWLINE_3: int = 3
    JAWLINE_4: int = 4
    JAWLINE_5: int = 5
    JAWLINE_6: int = 6
    JAWLINE_7: int = 7
    JAWLINE_8: int = 8
    JAWLINE_9: int = 9
    JAWLINE_10: int = 10
    JAWLINE_11: int = 11
    JAWLINE_12: int = 12
    JAWLINE_13: int = 13
    JAWLINE_14: int = 14
    JAWLINE_15: int = 15
    JAWLINE_16: int = 16
    LEFT_EYEBROW_17: int = 17
    LEFT_EYEBROW_18: int = 18
    LEFT_EYEBROW_19: int = 19
    LEFT_EYEBROW_20: int = 20
    LEFT_EYEBROW_21: int = 21
    RIGHT_EYEBROW_22: int = 22
    RIGHT_EYEBROW_23: int = 23
    RIGHT_EYEBROW_24: int = 24
    RIGHT_EYEBROW_25: int = 25
    RIGHT_EYEBROW_26: int = 26
    NOSE_BRIDGE_27: int = 27
    NOSE_BRIDGE_28: int = 28
    NOSE_BRIDGE_29: int = 29
    NOSE_BRIDGE_30: int = 30
    NOSE_BOTTOM_31: int = 31
    NOSE_BOTTOM_32: int = 32
    NOSE_BOTTOM_33: int = 33
    NOSE_BOTTOM_34: int = 34
    NOSE_BOTTOM_35: int = 35
    LEFT_EYE_36: int = 36
    LEFT_EYE_37: int = 37
    LEFT_EYE_38: int = 38
    LEFT_EYE_39: int = 39
    LEFT_EYE_40: int = 40
    LEFT_EYE_41: int = 41
    RIGHT_EYE_42: int = 42
    RIGHT_EYE_43: int = 43
    RIGHT_EYE_44: int = 44
    RIGHT_EYE_45: int = 45
    RIGHT_EYE_46: int = 46
    RIGHT_EYE_47: int = 47
    OUTER_LIPS_48: int = 48
    OUTER_LIPS_49: int = 49
    OUTER_LIPS_50: int = 50
    OUTER_LIPS_51: int = 51
    OUTER_LIPS_52: int = 52
    OUTER_LIPS_53: int = 53
    OUTER_LIPS_54: int = 54
    OUTER_LIPS_55: int = 55
    OUTER_LIPS_56: int = 56
    OUTER_LIPS_57: int = 57
    OUTER_LIPS_58: int = 58
    OUTER_LIPS_59: int = 59
    INNER_LIPS_60: int = 60
    INNER_LIPS_61: int = 61
    INNER_LIPS_62: int = 62
    INNER_LIPS_63: int = 63
    INNER_LIPS_64: int = 64
    INNER_LIPS_65: int = 65
    INNER_LIPS_66: int = 66
    INNER_LIPS_67: int = 67
