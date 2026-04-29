import { loadPyodide, type PyodideInterface } from 'pyodide';

const BASE_PACKAGE_PATH = '/assets/pyodide/';
// const MICROPIP = 'micropip-0.11.0-py3-none-any.whl';
const PACKAGES = [
      ['docopt', 'docopt-0.6.2-py2.py3-none-any.whl'],
      // ['pillow', 'pillow-11.3.0-cp313-cp313-pyodide_2025_0_wasm32.whl'],
      // ["psd-tools", 'psd_tools-1.16.0-cp313-cp313-pyodide_2025_0_wasm32.whl'],
      
      ["psd-tools", 'psd_tools-1.16.0-cp313-cp313-emscripten_4_0_9_wasm32.whl'],
      ['psdtosvg', 'psdtosvg-1.0.0-py3-none-any.whl'],
      ['potrace', 'potrace-1.0.0-py3-none-any.whl'],
];

export async function loadPyodideAndPackages(): Promise<PyodideInterface> {
  const pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/",
    // indexURL: BASE_PACKAGE_PATH,
    // packages: [MICROPIP]
    // packages: PACKAGES.map(([_, fileName]) => `${BASE_PACKAGE_PATH}${fileName}`)
  });
    await pyodide.loadPackage("micropip");

  const micropip = pyodide.pyimport("micropip");
  for (const [pkg, fileName] of PACKAGES) {
    console.log('loading package:', pkg);
    // await micropip.install(pkg);
    await micropip.install(`${BASE_PACKAGE_PATH}${fileName}`);
  }

  return pyodide;
}

export async function runpython(pyodide: PyodideInterface, file: File): Promise<string>  {
  const arrayBuffer = await file.arrayBuffer();
  pyodide.FS.writeFile('data.psd', new Uint8Array(arrayBuffer));
  return pyodide.runPython(`
    #import psd_tools
    import potrace
    print(1 + 2)
    #from psdtosvg import psd_file_to_svg
    #psd_file_to_svg('data.psd')
`) as string;
}

export async function loadAndRun(file: File) {
  const pyodide = await loadPyodideAndPackages();
  runpython(pyodide, file);
}