"""One picture of what a checkpoint transcribed: drawings in pairs, the record each was rendered
from beside the record the model wrote down.

The environment a run trains in has numpy but no image library, so the PNG is written here.
"""

import struct
import zlib

import numpy as np

# Between the tiles of one drawing, and between one drawing and the next.
GUTTER = 2
MARGIN = 10

BLANK = 255


def write(tiles, path, rows, across, per_drawing):
    """Lays `tiles` out `across` drawings per row and `rows` down and writes the picture to `path`.

    :param tiles: `(tile, width, height, colour)` pixels, the `per_drawing` tiles of one drawing
        one after the other.
    """
    tiles = np.asarray(tiles, dtype=np.uint8).transpose(0, 2, 1, 3)
    _, height, width, colours = tiles.shape
    drawing = per_drawing * width + (per_drawing - 1) * GUTTER
    picture = np.full(
        (rows * (height + MARGIN) - MARGIN, across * (drawing + MARGIN) - MARGIN, colours), BLANK, np.uint8
    )
    for at, tile in enumerate(tiles):
        row, column = divmod(at // per_drawing, across)
        top = row * (height + MARGIN)
        left = column * (drawing + MARGIN) + (at % per_drawing) * (width + GUTTER)
        picture[top:top + height, left:left + width] = tile
    _png(path, picture)


def _png(path, picture):
    """`picture` as an eight bit truecolour PNG."""
    height, width, _ = picture.shape
    scanlines = b"".join(b"\x00" + picture[row].tobytes() for row in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    with open(path, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")
        file.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        file.write(chunk(b"IDAT", zlib.compress(scanlines)))
        file.write(chunk(b"IEND", b""))
