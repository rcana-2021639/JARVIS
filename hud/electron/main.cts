/**
 * Proceso principal de Electron: la ventana del HUD.
 *
 * Sin marco, transparente (las esquinas cortadas dejan ver el escritorio),
 * siempre encima y arrastrable por su barra superior. En desarrollo carga el
 * servidor de Vite (JARVIS_HUD_URL); en produccion, dist/index.html.
 *
 * Seguridad: aislamiento de contexto, sandbox y sin Node en la pagina. El
 * preload expone solo cerrar y minimizar.
 */

import { BrowserWindow, app, ipcMain, screen } from "electron";
import * as fs from "node:fs";
import * as path from "node:path";

const DEV_URL = process.env["JARVIS_HUD_URL"];
/** Fraccion del lado menor de la pantalla que ocupa la ventana. */
const SIZE_RATIO = 0.55;
const MARGIN = 24;

function createWindow(): BrowserWindow {
  const area = screen.getPrimaryDisplay().workArea;
  const size = Math.round(Math.min(area.width, area.height) * SIZE_RATIO);
  const win = new BrowserWindow({
    width: size,
    height: size,
    x: area.x + area.width - size - MARGIN,
    y: area.y + MARGIN,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: true,
    hasShadow: false,
    backgroundColor: "#00000000",
    show: false,
    title: "JARVIS",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      // Oculta o minimizada, la ventana deja de pedir cuadros: el HUD se suspende.
      backgroundThrottling: true,
    },
  });
  // "floating" la mantiene sobre las ventanas normales sin tapar menus del sistema.
  win.setAlwaysOnTop(true, "floating");
  win.once("ready-to-show", () => win.show());
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event) => event.preventDefault());
  if (DEV_URL) {
    void win.loadURL(DEV_URL);
  } else {
    void win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
  return win;
}

/**
 * Autoprueba: con JARVIS_HUD_SELFTEST=segundos, el HUD corre ese tiempo, guarda
 * en JARVIS_HUD_REPORT (carpeta) los FPS de cada segundo, una captura y las
 * propiedades de la ventana, y se cierra. Es el banco de "60 fps durante 5 min".
 */
function scheduleSelfTest(win: BrowserWindow): void {
  const seconds = Number(process.env["JARVIS_HUD_SELFTEST"] ?? "0");
  const folder = process.env["JARVIS_HUD_REPORT"];
  if (!(seconds > 0) || folder === undefined) return;
  fs.mkdirSync(folder, { recursive: true });
  // Una captura cada 2,5 s: con el nucleo recorriendo estados, se ven los ocho.
  let shot = 0;
  const shots = setInterval(() => {
    void win.webContents.capturePage().then((image) => {
      shot += 1;
      fs.writeFileSync(path.join(folder, `shot-${String(shot).padStart(3, "0")}.png`), image.toPNG());
    });
  }, 2500);
  // Los FPS se guardan cada 10 s: si la ventana se cierra antes, no se pierden.
  const partial = setInterval(() => {
    void win.webContents
      .executeJavaScript("window.__jarvisFps ?? []")
      .then((fps: unknown) => fs.writeFileSync(path.join(folder, "fps-partial.json"), JSON.stringify(fps)));
  }, 10_000);
  win.on("close", () => {
    clearInterval(shots);
    clearInterval(partial);
  });
  setTimeout(() => {
    clearInterval(shots);
    clearInterval(partial);
    void (async () => {
      const fps: unknown = await win.webContents.executeJavaScript("window.__jarvisFps ?? []");
      const image = await win.webContents.capturePage();
      fs.mkdirSync(folder, { recursive: true });
      fs.writeFileSync(path.join(folder, "hud.png"), image.toPNG());
      const report = {
        seconds,
        fps,
        alwaysOnTop: win.isAlwaysOnTop(),
        transparentBackground: win.getBackgroundColor(),
        bounds: win.getBounds(),
        gpu: app.getGPUFeatureStatus(),
      };
      fs.writeFileSync(path.join(folder, "report.json"), JSON.stringify(report, null, 2));
      app.quit();
    })();
  }, seconds * 1000);
}

ipcMain.on("hud:close", (event) => BrowserWindow.fromWebContents(event.sender)?.close());
ipcMain.on("hud:minimize", (event) => BrowserWindow.fromWebContents(event.sender)?.minimize());

void app.whenReady().then(() => {
  scheduleSelfTest(createWindow());
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => app.quit());
