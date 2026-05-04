import { loadPyodide, type PyodideInterface } from 'pyodide';

const BASE_PACKAGE_PATH = './assets/pyodide/';

async function loadPyodideAndPackages(): Promise<PyodideInterface> {
  const pyodide = await loadPyodide({
    // indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/",
    indexURL: BASE_PACKAGE_PATH,
  });
    await pyodide.loadPackage("micropip");

  const micropip = pyodide.pyimport("micropip");
  await micropip.install('psdtosvg');

  return pyodide;
}

async function runpython(pyodide: PyodideInterface, file: File): Promise<string>  {
  const arrayBuffer = await file.arrayBuffer();
  pyodide.FS.writeFile('data.psd', new Uint8Array(arrayBuffer));
  const svg = await pyodide.runPythonAsync(`
    from psdtosvg import psd_file_to_svg
    psd_file_to_svg('data.psd')
`);
  return svg;
}

let PYODIDE_INSTANCE: PyodideInterface | null = null;
let PYODIDE_LOADING_PROMISE: Promise<PyodideInterface> | null = null;

export async function processFile(file: File) {
  if (!PYODIDE_INSTANCE) {
    if (!PYODIDE_LOADING_PROMISE) {
      PYODIDE_LOADING_PROMISE = loadPyodideAndPackages();
    }
    PYODIDE_INSTANCE = await PYODIDE_LOADING_PROMISE;
  }
  return await runpython(PYODIDE_INSTANCE, file);
}