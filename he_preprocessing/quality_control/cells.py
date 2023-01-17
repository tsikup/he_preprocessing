import os
import glob
import json

from dotmap import DotMap
from pprint import pprint


def check_cell_in_tile(tile_name, cell_path):
    """
    Check if cell is out of the tile's boundaries.

    Args:
        tile_name: Tile's filename.
        cell_path: Cell's filepath.

    Returns:
        boolean: True if cell in tile, else false
    """
    tile_coords = get_tile_coords(tile_name)
    cell_coords = get_cell_coords(cell_path)

    if tile_coords.x <= cell_coords.x <= tile_coords.x + tile_coords.w:
        if tile_coords.y <= cell_coords.y <= tile_coords.y + tile_coords.h:
            return True
    return False


def get_tile_coords(tile_name):
    """
    Get coordinates of a tile.

    Args:
        tile_name: Filename of the tile.

    Returns:
        DotMap(x=, y=)
    """
    tile_coords = tile_name.split('(')[1].split(')')[0].split(',')
    coords = {}
    for entry in tile_coords:
        key, value = entry.split('=')
        try:
            coords[key.strip()] = int(value.strip())
        except Exception:
            coords[key.strip()] = float(value.strip())
    return DotMap(coords)


def get_cell_coords(cell_path):
    """
    Get coordinates for the centorid of the cell's nucleus.

    Args:
        cell_path: Filepath for the cell.json file.

    Returns:
        DotMap(x=, y=)
    """
    with open(cell_path) as f:
        cell = DotMap(json.load(f))
    return cell.nucleusCentroid
