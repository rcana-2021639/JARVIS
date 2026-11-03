/**
 * Lo unico que la pagina puede pedirle a Electron: cerrar y minimizar la ventana.
 */

import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("jarvisWindow", {
  close: (): void => ipcRenderer.send("hud:close"),
  minimize: (): void => ipcRenderer.send("hud:minimize"),
});
