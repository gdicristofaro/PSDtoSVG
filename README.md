Converts a Photoshop file (PSD) to an Scalable Vector Graphics (SVG) file. All Layers in PSD file except for bottom-most layer are traced using pypotrace. The paths determined by pypotrace are used to determine the SVG paths in the SVG.

Project includes potrace.py. This file is a port from [Kilobyte's potrace library](https://github.com/kilobtye/potrace) which in turn was ported from the [potrace library created by Peter Selinger](https://potrace.sourceforge.net/).

Feel free to try the [demo](https://gdicristofaro.github.io/PSDtoSVG/)

The `src` directory contains the python package source to build `PSDtoSVG` (which can be built with `python -m build` after installing python build tools).

The `flask` directory contains the basic flask app and can be built using the `requirements.txt` file.  Note that this must be done after building the `PSDtoSVG` package in `src`.

The `pyodide` directory contains a web application that runs the python code in browser using pyodide.