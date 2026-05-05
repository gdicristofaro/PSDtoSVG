Converts a Photoshop file (PSD) to an Scalable Vector Graphics (SVG) file. All Layers in PSD file except for bottom-most layer are traced using pypotrace. The paths determined by pypotrace are used to determine the SVG paths in the SVG.

Project includes `potrace.py`. This file is a port from [Kilobyte's potrace library](https://github.com/kilobtye/potrace) which in turn was ported from the [potrace library created by Peter Selinger](https://potrace.sourceforge.net/).

**Feel free to try the [demo](https://gdicristofaro.github.io/PSDtoSVG/)**

# Project structure:
- `src`: the python package source to build `PSDtoSVG`.
- `frontend`: contains a web application that runs the python code in browser using pyodide or can be built to send requests to convert PSD files to a backend.
- `flask`: basic flask app requiring the `PSDtoSVG` build and the `frontend` build.

