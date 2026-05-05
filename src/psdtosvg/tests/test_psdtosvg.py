import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from psdtosvg.potrace import Bitmap
from psdtosvg.psdtosvg import (
    get_bitmap_arr,
    avg_color,
    svg_converter,
    gather_layers,
    handle_layers,
    get_svg,
    psd_to_svg,
    psd_file_to_svg,
    psd_stream_to_svg
)


class TestPsdToSvg(unittest.TestCase):

    def test_get_bitmap_arr(self):
        pixel_array = np.array([
            [255, 0, 0, 255],
            [0, 255, 0, 0]
        ])
        bitmap = get_bitmap_arr(pixel_array, 2, 1, 3)
        self.assertIsInstance(bitmap, Bitmap)
        self.assertEqual(bitmap.w, 2)
        self.assertEqual(bitmap.h, 1)
        self.assertEqual(bitmap.data, [True, False])

    def test_avg_color(self):
        pixel_array = np.array([
            [100, 50, 25, 255],
            [200, 150, 75, 255],
            [0, 0, 0, 0]  # Alpha 0, should be ignored
        ])
        color = avg_color(pixel_array)
        self.assertEqual(color['red'], 150)
        self.assertEqual(color['green'], 100)
        self.assertEqual(color['blue'], 50)

    def test_gather_layers(self):
        class MockPSDImage:
            def __init__(self, children):
                self.children = children
            def __iter__(self):
                return iter(self.children)

        class MockGroup: pass
        class MockLayer: pass
        
        with patch('psdtosvg.psdtosvg.PSDImage', MockPSDImage), \
             patch('psdtosvg.psdtosvg.Group', MockGroup), \
             patch('psdtosvg.psdtosvg.Layer', MockLayer):
             
             layer1 = MockLayer()
             layer2 = MockLayer()
             group = MockGroup()
             group.group_layers = [layer2]
             
             psd = MockPSDImage([layer1, group])
             
             layers = gather_layers(psd)
             self.assertEqual(layers, [layer1, layer2])

    @patch('psdtosvg.psdtosvg.get_bitmap_arr')
    @patch('psdtosvg.psdtosvg.process')
    @patch('psdtosvg.psdtosvg.get_svg_path')
    @patch('psdtosvg.psdtosvg.avg_color')
    def test_svg_converter_path(self, mock_avg_color, mock_get_svg_path, mock_process, mock_get_bitmap_arr):
        layer = MagicMock()
        layer.bbox = (0, 0, 10, 10)
        layer.numpy.return_value = np.zeros((10, 10, 4))
        layer.name = "Layer 1"
        
        mock_process.return_value = [MagicMock(curve="mock_curve")]
        mock_get_svg_path.return_value = "M 0 0"
        mock_avg_color.return_value = {'red': 0, 'green': 0, 'blue': 0}
        
        result = svg_converter(layer, 1)
        self.assertEqual(result['id'], "Layer1_1")
        self.assertEqual(result['svg_paths'], "M 0 0")
        
    def test_svg_converter_empty(self):
        layer = MagicMock()
        layer.bbox = (0, 0, 0, 0) # empty layer
        self.assertEqual(svg_converter(layer, 1), {})
        
    def test_svg_converter_dataurl(self):
        layer = MagicMock()
        layer.bbox = (0, 0, 10, 10)
        layer.numpy.return_value = np.zeros((10, 10, 4))
        layer.name = "Layer 1"
        
        result = svg_converter(layer, 1, get_dataurl=True)
        self.assertIn('image', result)
        self.assertTrue(result['image'].startswith('data:image/png;base64,'))
        self.assertEqual(result['id'], "Layer1_1")

    @patch('psdtosvg.psdtosvg.gather_layers')
    @patch('psdtosvg.psdtosvg.svg_converter')
    def test_handle_layers(self, mock_svg_converter, mock_gather_layers):
        mock_gather_layers.return_value = ['layer1', 'layer2']
        mock_svg_converter.side_effect = [{'id': 'img1'}, {'id': 'path1'}]
        
        result = handle_layers('psd_mock')
        
        self.assertEqual(len(result), 2)
        mock_svg_converter.assert_any_call('layer1', 0, True) # first layer gets get_dataurl=True
        mock_svg_converter.assert_any_call('layer2', 1)

    def test_get_svg(self):
        layers = [
            {'image': 'data:image/png;base64,123', 'id': 'img1', 'x': 0, 'y': 0, 'width': 10, 'height': 10},
            {'svg_paths': 'M 0 0 L 10 10 Z', 'color': {'red': 255, 'green': 0, 'blue': 0}, 'id': 'path1'}
        ]
        svg = get_svg(layers, 100, 100)
        self.assertIn('<svg', svg)
        self.assertIn('viewBox="0 0 100 100"', svg)
        self.assertIn('xlink:href="data:image/png;base64,123"', svg)
        self.assertIn('d="M 0 0 L 10 10 Z"', svg)
        self.assertIn('rgb(255,0,0)', svg)
        self.assertIn('</svg>', svg)

    @patch('psdtosvg.psdtosvg.handle_layers')
    @patch('psdtosvg.psdtosvg.get_svg')
    def test_psd_to_svg(self, mock_get_svg, mock_handle_layers):
        mock_handle_layers.return_value = ['layer_data']
        mock_get_svg.return_value = '<svg></svg>'
        
        psd = MagicMock()
        psd.width = 100
        psd.height = 100
        
        result = psd_to_svg(psd)
        
        self.assertEqual(result, '<svg></svg>')
        mock_get_svg.assert_called_with(['layer_data'], 100, 100)

    @patch('psdtosvg.psdtosvg.PSDImage.open')
    @patch('psdtosvg.psdtosvg.psd_to_svg')
    def test_psd_file_to_svg(self, mock_psd_to_svg, mock_open):
        mock_open.return_value = 'psd_mock'
        mock_psd_to_svg.return_value = '<svg></svg>'
        
        result = psd_file_to_svg('path/to/file.psd')
        
        self.assertEqual(result, '<svg></svg>')
        mock_open.assert_called_with('path/to/file.psd')

    @patch('psdtosvg.psdtosvg.PSDImage.open')
    @patch('psdtosvg.psdtosvg.psd_to_svg')
    def test_psd_stream_to_svg(self, mock_psd_to_svg, mock_open):
        mock_open.return_value = 'psd_mock'
        mock_psd_to_svg.return_value = '<svg></svg>'
        
        result = psd_stream_to_svg('stream_mock')
        
        self.assertEqual(result, '<svg></svg>')
        mock_open.assert_called_with('stream_mock')


if __name__ == '__main__':
    unittest.main()
