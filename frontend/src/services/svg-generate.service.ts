export async function processFile(file: File): Promise<string> {
const module = await (import.meta.env.VITE_BUILD_BACKEND
          ? import('../services/svg-generate.service.backend')
          : import('../services/svg-generate.service.pyodide'));
        const svgString = await module.processFile(file);
        return svgString;
}