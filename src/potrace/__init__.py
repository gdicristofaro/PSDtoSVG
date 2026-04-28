"""
potrace - A Python port of the Potrace algorithm for tracing bitmaps to SVG paths.

This package provides functionality to convert bitmap images into SVG path data
using the Potrace algorithm.

Main classes:
    - Point: Defines a point with x,y coordinates
    - Bitmap: Defines image data as a sequence of boolean values
    - Path: The path structure for the Potrace algorithm
    - Curve: Defines elements of the curve

Main functions:
    - process: Handles processing bitmap image, converting image to path outlines
    - get_svg_path: Converts curve to SVG path
    - get_svg: Gets the full SVG string with all paths included
"""

from .potrace import (
    Point,
    Bitmap,
    Path,
    Curve,
    process,
    get_svg_path,
    get_svg,
)

__all__ = [
    'Point',
    'Bitmap',
    'Path',
    'Curve',
    'process',
    'get_svg_path',
    'get_svg',
]

__version__ = '1.0.0'