"""One picture of what a checkpoint transcribed: drawings in pairs, the record each was rendered
from beside the record the model wrote down.

The environment a run trains in has numpy but no image library, so the PNG is written here.
"""

import struct
import zlib

import numpy as np

# Between the two drawings of a pair, and between one pair and the next.
GUTTER = 2
MARGIN = 10

BLANK = 255


def write(tiles, path, rows, pairs):
    """Lays `tiles` out `pairs` across and `rows` down and writes the picture to `path`.

    :param tiles: `(tile, width, height, colour)` pixels, every record followed by the
        transcription of the same drawing.
    """
    tiles = np.asarray(tiles, dtype=np.uint8).transpose(0, 2, 1, 3)
    _, height, width, colours = tiles.shape
    pair = 2 * width + GUTTER
    picture = np.full(
        (rows * (height + MARGIN) - MARGIN, pairs * (pair + MARGIN) - MARGIN, colours), BLANK, np.uint8
    )
    for at, tile in enumerate(tiles):
        row, column = divmod(at // 2, pairs)
        top = row * (height + MARGIN)
        left = column * (pair + MARGIN) + (at % 2) * (width + GUTTER)
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
