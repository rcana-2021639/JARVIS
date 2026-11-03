import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// La ventana de Electron carga el resultado con file://, por eso las rutas son relativas.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173, strictPort: true, host: "127.0.0.1" },
  build: { outDir: "dist", target: "es2022", chunkSizeWarningLimit: 1500 },
});
