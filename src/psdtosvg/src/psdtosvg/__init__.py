"""
psdtosvg - A Python package for converting PSD files to SVG format.

This package provides functionality to convert Adobe Photoshop PSD files
into SVG format, using the potrace algorithm for path tracing.

Main functions:
    - psd_to_svg: Main method for converting a PSD object to SVG
    - psd_file_to_svg: Converts a PSD file path to an SVG string
    - psd_stream_to_svg: Converts a PSD stream to an SVG string
    - get_svg: Creates an SVG string with all layers included
    - handle_layers: Analyzes PSD and converts layers to SVG items
    - svg_converter: Converts a layer into an SVG readable item
    - gather_layers: Gathers all layers recursively from PSD groups
    - get_bitmap_arr: Creates bitmap array from image data
    - avg_color: Gets the average color from image data
"""

from .psdtosvg import (
    psd_to_svg,
    psd_file_to_svg,
    psd_stream_to_svg,
    get_svg,
    handle_layers,
    svg_converter,
    gather_layers,
    get_bitmap_arr,
    avg_color,
    STROKE_WIDTH,
    FILL_OPACITY,
)

__all__ = [
    'psd_to_svg',
    'psd_file_to_svg',
    'psd_stream_to_svg',
    'get_svg',
    'handle_layers',
    'svg_converter',
    'gather_layers',
    'get_bitmap_arr',
    'avg_color',
    'STROKE_WIDTH',
    'FILL_OPACITY',
]

__version__ = '1.0.0'