# Frontend

This is the React frontend for running PSDtoSVG.  It can be run in two modes: an entirely frontend application using pyodide to run python in browser or with a server backend (the [`flask`](../flask/) directory) if built with `npm run build-server-frontend`.

# Pyodide notes

See [`src/services/svg-generate.service.pyodide.ts`](./src/services/svg-generate.service.pyodide.ts) as well as the [`public/assets/pyodide` directory](./public/assets/pyodide) for more information, but the overview is as follows.  Pyodide can be run by loading required files from a remote repository with something like 

```
const pyodide = await loadPyodide({
  indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/",
});
```

or from relative paths with something like:

```
const pyodide = await loadPyodide({
  indexURL: "public/assets/pyodide",
});
```
Currently, the project loads pyodide resources from relative paths to avoid possible remote resource drift.

## Building

Pyodide dependencies/wheels can be built with [pyodide build](https://pypi.org/project/pyodide-build/) or downloaded remotely from pypi or jsdelivr.  In order to run properly, the wheel must not have native dependencies or must be built using pyodide build to properly convert to WebAssembly.  In order to properly get the resources, I built `psd-tools` with pyodide build as it had native dependencies.  I built PSDtoSVG using the normal `python -m build` command.  I downloaded the rest of the files from a relative reference in `https://cdn.jsdelivr.net/pyodide/v0.29.3/full/`.  There are some packages like Pillow that have a specific pyodide generated build.  At the time of writing, there is an issue with wheel naming conventions (see [https://github.com/pyodide/pyodide/issues/6177](https://github.com/pyodide/pyodide/issues/6177) for more information).  The package name is parsed to determine required emscripten version.  To avoid this issue, I renamed natively created packages as they were built with the same version of pyodide running the wheel.

## `pyodide-lock.json`

After gathering all the necessary wheels, I generated the `pyodide-lock.json`.  The important piece here is to ensure the file name and the hash (i.e. `sha256sum <wheel file>`) are correct.  This lock file will instruct micropip on the file names available as well as allow it to do a proper check on the hashes.

# Pyodide important links

- [https://verbitskiy.co/blog/no-backend-needed-running-python-in-react-with-pyodide/](https://verbitskiy.co/blog/no-backend-needed-running-python-in-react-with-pyodide/)
- [https://pyodide.org/en/stable/usage/loading-packages.html](https://pyodide.org/en/stable/usage/loading-packages.html)
- [https://pyodide.org/en/stable/usage/loading-custom-python-code.html](https://pyodide.org/en/stable/usage/loading-custom-python-code.html)
- [https://pyodide.org/en/stable/development/building-packages-from-source.html](https://pyodide.org/en/stable/development/building-packages-from-source.html)
- [https://pyodide.org/en/stable/development/building-from-sources.html](https://pyodide.org/en/stable/development/building-from-sources.html)
- [https://github.com/pyodide/pyodide/issues/6177](https://github.com/pyodide/pyodide/issues/6177)
- [https://pyodide.org/en/stable/console.html](https://pyodide.org/en/stable/console.html)
- [https://pypi.org/project/pyodide-build/#history](https://pypi.org/project/pyodide-build/#history)
- [https://emscripten.org/docs/getting_started/downloads.html](https://emscripten.org/docs/getting_started/downloads.html)

