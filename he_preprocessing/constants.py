TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

SCALE_FACTOR = 32
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

FONT_PATH = None

IMAGE_EXT = "png"
THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = "jpg"

CELL_COLORS = {
    "Tumor": {
        "cname": "red",
        "r_channel": 255,
        "g_channel": 0,
        "b_channel": 0,
    },
    "Stroma": {
        "cname": "green",
        "r_channel": 0,
        "g_channel": 128,
        "b_channel": 0,
    },
    "Immune cells": {
        "cname": "purple",
        "r_channel": 128,
        "g_channel": 0,
        "b_channel": 128,
    },
    "Other": {
        "cname": "yellow",
        "r_channel": 255,
        "g_channel": 255,
        "b_channel": 0,
    },
}


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
