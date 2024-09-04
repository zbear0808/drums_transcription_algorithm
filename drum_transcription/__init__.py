__version__ = "0.1.0"

GROOVE_PITCH_NAMES = {
    36: (0, "Kick"),
    38: (1, "Snare_Head"),
    40: (1, "Snare_Rim"),
    37: (1, "Snare_X-Stick"),
    48: (2, "Tom1"),
    50: (2, "Tom1_Rim"),
    45: (3, "Tom2"),
    47: (3, "Tom2_Rim"),
    43: (4, "Tom3_Head"),
    58: (4, "Tom3_Rim"),
    46: (5, "HHOpen_Bow"),
    26: (5, "HHOpen_Edge"),
    42: (6, "HHClosed_Bow"),
    22: (6, "HHClosed_Edge"),
    44: (6, "HHPedal"),
    49: (7, "Crash1_Bow"),
    55: (7, "Crash1_Edge"),
    57: (7, "Crash2_Bow"),
    52: (7, "Crash2_Edge"),
    51: (8, "Ride_Bow"),
    59: (8, "Ride_Edge"),
    53: (8, "Ride_Bell"),
}

GROOVE_PITCH_POST_PROCESS = {
    0: (36, "Kick"),
    1: (38, "Snare_Head"),
    2: (48, "Tom1"),
    3: (45, "Tom2"),
    4: (43, "Tom3_Head"),
    5: (46, "HHOpen_Bow"),
    6: (42, "HHClosed_Bow"),
    7: (49, "Crash1_Bow"),
    8: (51, "Ride_Bow"),
}