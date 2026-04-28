import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from "vite-plugin-static-copy";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

// pyodide setup taken from: https://pyodide.org/en/latest/usage/working-with-bundlers.html

const PYODIDE_PACKAGES = new Set(['micropip', 'pillow', 'psd-tools']);

const pyodideDirName = "pyodide";

export function viteStaticCopyPyodide() {
  const pyodideDir = dirname(fileURLToPath(import.meta.resolve(pyodideDirName)));
  return viteStaticCopy({
  targets: [
    // Copy only the specific .whl files you need
    {
      src: [
        join(pyodideDir, 'micropip-*-py3-none-any.whl'),
        // add other specific wheel files here
      ],
      dest: 'assets/pyodide',
    },
    // Transform lockfile to only include selected packages
    {
      src: join(pyodideDir, 'pyodide-lock.json'),
      dest: 'assets/pyodide',
      transform: (content) => {
        const lockfile = JSON.parse(content.toString());
        lockfile.packages = Object.fromEntries(
          Object.entries(lockfile.packages).filter(([name]) =>
            PYODIDE_PACKAGES.has(name)
          )
        );
        return JSON.stringify(lockfile);
      },
    },
  ],
});
}

export default defineConfig({
  optimizeDeps: { exclude: [pyodideDirName] },
  plugins: [viteStaticCopyPyodide(), react()],
});