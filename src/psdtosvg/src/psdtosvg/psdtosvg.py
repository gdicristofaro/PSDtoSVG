from __future__ import annotations
from psd_tools import PSDImage
from psd_tools.api.layers import Group, Layer
from typing import Union, List, Dict, Any
import base64
from io import BytesIO
import re
import numpy as np
from PIL import Image

from psdtosvg.potrace import Bitmap, process, get_svg_path


def get_bitmap_arr(pixel_array: np.ndarray, width: int, height: int, alpha_channel: int) -> Bitmap:
    """
    based on image data, creates a list where pixel is located at
    [x + width * y], the pixel is either true or false
    :param pixel_array: the image data as an array of colors
    :param width: the width of the image
    :param height: the height of the image
    :param alpha_channel: which index in the color item contains the alpha
    :returns: a list of pixels to be consumed by potrace
    """
    data = [pixel[alpha_channel] > 0 for pixel in pixel_array]
    return Bitmap(width, height, data)


def avg_color(pixel_array: np.ndarray) -> dict:
    """
    gets the average color found in the image data
    :param pixel_array: the image data as an array of colors
    :returns: the average color found in the image
    """
    avg = np.mean(pixel_array[pixel_array[:, 3] > 0], axis=0)
    return {
        'red': avg[0],
        'green': avg[1],
        'blue': avg[2]
    }


def svg_converter(layer: Layer, id_num: int, get_dataurl: bool = False) -> dict:
    """
    converts a layer into an svg readable item
    :param layer: the PSD image layer to be converted
    :param id_num: the index of the PSD image layer for svg id naming purposes
    :param get_dataurl: whether or not the layer should be converted to an
    image dataurl that can be used as an image in an svg
    """
    # get metrics for layer
    # print layer
    x_offset = layer.bbox[0]
    y_offset = layer.bbox[1]
    width = layer.bbox[2] - layer.bbox[0]
    height = layer.bbox[3] - layer.bbox[1]

    if width <= 0 or height <= 0:
        return {}

    # get alpha channel
    psd_numpy = layer.numpy()
    img_dat = np.clip(psd_numpy * 255, 0, 255).astype('uint8')
    pixel_array = img_dat.reshape(-1, img_dat.shape[-1])

    layer_id = str(re.sub(r'\W+', '', "%s_%d" % (layer.name, id_num)))

    # cannot convert image or should be dataurl anyway, convert to image
    if len(img_dat[0]) < 4 or get_dataurl:
        # taken from https://stackoverflow.com/questions/42503995/
        # how-to-get-a-pil-image-as-a-base64-encoded-string/42504858
        buffer = BytesIO()
        
        image = Image.fromarray(img_dat, 'RGBA')
        image.save(buffer, format="PNG")
        buffer.seek(0)
        img_bytes = buffer.read()
        base64_bytes = base64.b64encode(img_bytes)
        base64_str = base64_bytes.decode('ascii')
        # print("base64 is " + base64_str[:30] + "... (" + str(len(base64_str)) + " characters)")

        # base64 string from https://en.wikipedia.org/wiki/wiki/Data_URI_scheme
        return {
            'image': "data:image/png;base64," + base64_str,
            'x': x_offset,
            'y': y_offset,
            'width': width,
            'height': height,
            'id': layer_id
        }
    else:
        # create array to analyze with pypotrace
        po_data = get_bitmap_arr(pixel_array, width, height, 3)
        pathlist = [get_svg_path(p.curve, x_offset, y_offset)
                    for p in process(po_data, optcurve=False)]
        joined_paths = ' '.join(pathlist)

        return {
            'svg_paths': joined_paths,
            'color': avg_color(pixel_array),
            'id': layer_id
        }


def gather_layers(item: Union[PSDImage, Group, Layer]) -> list:
    """
    gathers all layers (recursively checking in groups) and
    returns a list of layers to be converted to svg items
    :param item: an item relating to a PSD image (i.e. the PSD
    image, a PSD group, or a PSD layer)
    :returns: a list of layers
    """
    if isinstance(item, PSDImage):
        layer_list = []
        for child in item:
            child_layers = gather_layers(child)
            layer_list.extend(child_layers)
        return layer_list
    elif isinstance(item, Group):
        layer_list = []
        for child in item.group_layers:
            child_layers = gather_layers(child)
            layer_list.extend(child_layers)
        return layer_list
    elif isinstance(item, Layer):
        return [item]


def handle_layers(psd: PSDImage) -> list:
    """
    analyzes psd and converts all pertinent aspects to something
    that will be converted to an svg
    :param psd: the psd image
    :returns: a list of svg path or dataurl items to be added to the svg
    """
    layer_list = gather_layers(psd)

    svg_strs = []
    for index, layer in enumerate(layer_list):

        # last image is left as image and not converted to svg
        if index == 0: #len(layer_list) - 1:
            svg_strs.append(svg_converter(layer, index, True))
        else:
            svg_strs.append(svg_converter(layer, index))

    return svg_strs


STROKE_WIDTH = 2
FILL_OPACITY = .6


def get_svg(all_layers: list, width: int, height: int) -> str:
    """
    creates an SVG file with all layers included
    :param all_layers: the processed layer items to be added
    :param width: the width of the svg viewbox
    :param height: the height of the svg viewbox
    :returns: the generated SVG string
    """

    svg_str = '''<?xml version="1.0" encoding="UTF-8" ?>
<svg
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   viewBox="0 0 %d %d"
   version="1.1">''' % (width, height)

    for g in all_layers:
        if 'image' in g:
            svg_str += (('<image class="%s" xlink:href="%s" height="%d"' +
                         ' width="%d" x="%d" y="%d"/>\n') %
                        (g['id'], g['image'], g['height'],
                         g['width'], g['x'], g['y']))

        elif 'svg_paths' in g:
            color = g['color']
            rgb_portion = ("%d,%d,%d" % (color['red'], color['green'],
                                         color['blue']))

            stroke = ("rgb(%s)" % rgb_portion)
            fill = ("rgba(%s,%f)" % (rgb_portion, FILL_OPACITY))
            path = g['svg_paths']
            this_id = g['id']

            svg_str += (('<path class="%s" d="%s" fill="%s" stroke="%s" ' +
                         'stroke-width="%d"/>\n') % (this_id, path, fill,
                                                     stroke, STROKE_WIDTH))

    return svg_str + '</svg>'


def psd_to_svg(psd: PSDImage) -> str:
    """
    main method for converting psd to an svg item
    :param psd: the psd file
    :returns: the svg string
    """
    all_groups = handle_layers(psd)
    width = psd.width
    height = psd.height
    return get_svg(all_groups, width, height)


def psd_file_to_svg(psd_path: str) -> str:
    """
    converts a psd file path to an svg
    :param psd_path: the path to the psd
    :returns: the svg string
    """
    psd = PSDImage.open(psd_path)
    svg = psd_to_svg(psd)
    return svg


def psd_stream_to_svg(psd_stream: BytesIO) -> str:
    """
    converts a psd file path to an svg
    :param psd_stream: the psd stream
    :returns: the svg string
    """
    psd = PSDImage.open(psd_stream)
    return psd_to_svg(psd)