# psdtosvg

A Python package for converting PSD (Photoshop Document) files to SVG format.  The bottom layer is left as an image, and all others are coverted to svg shapes.

## Development

### Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -e ".[dev]"
```

### Running tests

```bash
python -m unittest discover -s tests
```

### Building the package

```bash
pip install build
python -m build
```

The built distributions will be placed in the `dist/` directory.

