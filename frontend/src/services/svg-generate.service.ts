export async function processFile(file: File): Promise<string> {
    console.log("processing result: ", import.meta.env.VITE_BUILD_BACKEND);
const module = await (import.meta.env.VITE_BUILD_BACKEND
          ? import('../services/svg-generate.service.backend')
          : import('../services/svg-generate.service.pyodide'));
        const svgString = await module.processFile(file);
        return svgString;
}