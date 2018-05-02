Converts a Photoshop file (PSD) to an Scalable Vector Graphics (SVG) file. All Layers in PSD file except for bottom-most layer are traced using pypotrace. The paths determined by pypotrace are used to determine the SVG paths in the SVG.

Project includes potrace.py. This file is a port from [Kilobyte's potrace library](https://github.com/kilobtye/potrace) which in turn was ported from the [potrace library created by Peter Selinger](https://potrace.sourceforge.net/).

Feel free to try the [demo on heroku](https://psdtosvg.herokuapp.com/)


This can be run after installing dependencies (which can be found in requirements.txt):
```
pip install psd-tools
pip install pillow
pip install flask
```

and then running:
```
python psdtosvg.py
```

with default configuration of flask, this runs on localhost:5000.