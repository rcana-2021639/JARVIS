/**
 * La ventana del HUD en la Fase 4: solo el nucleo, perfecto.
 *
 * Un marco de esquinas cortadas con una barra minima para arrastrar la ventana
 * y, abajo, el estado en texto: el indicador inequivoco de si Jarvis duerme,
 * escucha o procesa. Los paneles son de la Fase 5.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Reactor } from "./core3d/Reactor";
import { type CoreSnapshot, coreLink } from "./services/core";

declare global {
  interface Window {
    /** Lo que expone el preload de Electron. Ausente en el navegador. */
    jarvisWindow?: { close: () => void; minimize: () => void };
  }
}

const SHOW_FPS = import.meta.env.DEV || new URLSearchParams(window.location.search).has("fps");

const LINK_LABEL: Record<CoreSnapshot["status"], string> = {
  conectado: "NUCLEO EN LINEA",
  reconectando: "BUSCANDO NUCLEO",
  simulado: "SIMULACION",
};

export function App(): JSX.Element {
  const [snapshot, setSnapshot] = useState<CoreSnapshot>(coreLink.current());
  const fpsText = useRef<HTMLSpanElement>(null);
  const lodText = useRef<HTMLSpanElement>(null);

  useEffect(() => coreLink.onStatus(setSnapshot), []);
  // El contador se escribe directo en el DOM: un setState por segundo es poco,
  // pero no hace falta re-renderizar nada por el.
  const onFps = useCallback((fps: number) => {
    if (fpsText.current) fpsText.current.textContent = `${fps.toFixed(0)} FPS`;
  }, []);
  const onLod = useCallback((particles: number) => {
    if (lodText.current) lodText.current.textContent = `${(particles / 1000).toFixed(0)}K PTS`;
  }, []);

  return (
    <div className="frame" data-state={snapshot.state}>
      <svg className="frame__edge" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        <polygon points="3,0 97,0 100,3 100,97 97,100 3,100 0,97 0,3" vectorEffect="non-scaling-stroke" />
      </svg>
      <header className="frame__bar">
        <span className="frame__title">J.A.R.V.I.S</span>
        <span className={`frame__link frame__link--${snapshot.status}`}>{LINK_LABEL[snapshot.status]}</span>
        {window.jarvisWindow ? (
          <span className="frame__controls">
            <button type="button" onClick={() => window.jarvisWindow?.minimize()} aria-label="Minimizar">
              —
            </button>
            <button type="button" onClick={() => window.jarvisWindow?.close()} aria-label="Cerrar">
              ×
            </button>
          </span>
        ) : null}
      </header>
      <main className="frame__stage">
        <Reactor onFps={onFps} onLod={onLod} />
      </main>
      <footer className="frame__status">
        <span className="frame__state">{snapshot.state}</span>
        {SHOW_FPS ? (
          <span className="frame__perf">
            <span ref={fpsText}>-- FPS</span>
            <span ref={lodText} />
          </span>
        ) : null}
      </footer>
    </div>
  );
}
