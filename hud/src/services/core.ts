/**
 * La conexion con el nucleo Python.
 *
 * Cliente WebSocket al servidor local del nucleo (127.0.0.1) con reconexion
 * automatica y retroceso. Si el nucleo no esta corriendo, el HUD no se queda
 * muerto: pasa a una SIMULACION que recorre los ocho estados con audio
 * sintetico, para poder trabajar el HUD sin levantar todo. En cuanto el nucleo
 * aparece, la simulacion se apaga sola.
 */

import { BAND_COUNT, type CoreMessage, type CoreState, parseMessage } from "./protocol";

export type LinkStatus = "conectado" | "reconectando" | "simulado";

export interface CoreSnapshot {
  readonly state: CoreState;
  readonly status: LinkStatus;
}

type Listener = (message: CoreMessage) => void;
type StatusListener = (snapshot: CoreSnapshot) => void;

const DEFAULT_URL = "ws://127.0.0.1:8766";
const MAX_BACKOFF_MS = 8000;
/** Sin nucleo durante este tiempo, arranca la simulacion; la reconexion sigue intentando. */
const SIMULATE_AFTER_MS = 1500;

function coreUrl(): string {
  return new URLSearchParams(window.location.search).get("core") ?? DEFAULT_URL;
}

export class CoreLink {
  private socket: WebSocket | null = null;
  private backoff = 500;
  private readonly listeners = new Set<Listener>();
  private readonly statusListeners = new Set<StatusListener>();
  private simulation: Simulation | null = null;
  private simulateTimer: number | null = null;
  private retryTimer: number | null = null;
  private closed = true;
  private snapshot: CoreSnapshot = { state: "DORMIDO", status: "reconectando" };

  constructor(private readonly url: string = coreUrl()) {}

  start(): void {
    if (!this.closed) return;
    this.closed = false;
    this.connect();
    this.armSimulation();
  }

  stop(): void {
    this.closed = true;
    this.socket?.close();
    this.stopSimulation();
    this.clearTimers();
  }

  onMessage(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  onStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.snapshot);
    return () => {
      this.statusListeners.delete(listener);
    };
  }

  current(): CoreSnapshot {
    return this.snapshot;
  }

  private connect(): void {
    if (this.closed) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url);
    } catch {
      this.scheduleRetry();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      this.backoff = 500;
      this.stopSimulation();
      if (this.simulateTimer !== null) window.clearTimeout(this.simulateTimer);
      this.setStatus("conectado");
    };
    socket.onmessage = (event: MessageEvent<unknown>) => {
      if (typeof event.data !== "string") return;
      const message = parseMessage(event.data);
      if (message !== null) this.dispatch(message);
    };
    socket.onclose = () => {
      this.socket = null;
      if (this.closed) return;
      if (this.snapshot.status === "conectado") {
        this.setStatus("reconectando");
        this.armSimulation();
      }
      this.scheduleRetry();
    };
    socket.onerror = () => socket.close();
  }

  private armSimulation(): void {
    if (this.simulateTimer !== null) window.clearTimeout(this.simulateTimer);
    this.simulateTimer = window.setTimeout(() => this.startSimulation(), SIMULATE_AFTER_MS);
  }

  private scheduleRetry(): void {
    if (this.closed) return;
    this.retryTimer = window.setTimeout(() => this.connect(), this.backoff);
    this.backoff = Math.min(MAX_BACKOFF_MS, this.backoff * 2);
  }

  private clearTimers(): void {
    if (this.simulateTimer !== null) window.clearTimeout(this.simulateTimer);
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    this.simulateTimer = null;
    this.retryTimer = null;
  }

  private dispatch(message: CoreMessage): void {
    if (message.type === "state" || message.type === "hello") {
      this.snapshot = { ...this.snapshot, state: message.state };
      this.statusListeners.forEach((listener) => listener(this.snapshot));
    }
    this.listeners.forEach((listener) => listener(message));
  }

  private setStatus(status: LinkStatus): void {
    this.snapshot = { ...this.snapshot, status };
    this.statusListeners.forEach((listener) => listener(this.snapshot));
  }

  private startSimulation(): void {
    if (this.closed || this.simulation !== null || this.snapshot.status === "conectado") return;
    this.setStatus("simulado");
    this.simulation = new Simulation((message) => this.dispatch(message));
    this.simulation.start();
  }

  private stopSimulation(): void {
    this.simulation?.stop();
    this.simulation = null;
  }
}

interface ScriptStep {
  readonly state: CoreState;
  readonly ms: number;
  readonly audio: "mic" | "tts" | null;
  readonly load: number;
}

/** El guion de la simulacion: cada estado, cuanto dura y que audio lleva. */
const SCRIPT: readonly ScriptStep[] = [
  { state: "DORMIDO", ms: 6000, audio: null, load: 0 },
  { state: "DESPIERTO", ms: 900, audio: null, load: 0.1 },
  { state: "ESCUCHANDO", ms: 3500, audio: "mic", load: 0.1 },
  { state: "PENSANDO", ms: 2800, audio: null, load: 0.9 },
  { state: "HABLANDO", ms: 4500, audio: "tts", load: 0.3 },
  { state: "EJECUTANDO", ms: 3000, audio: null, load: 0.5 },
  { state: "DORMIDO", ms: 3000, audio: null, load: 0 },
  { state: "ERROR", ms: 1600, audio: null, load: 0.2 },
  { state: "DORMIDO", ms: 2500, audio: null, load: 0 },
  { state: "CONFIRMANDO", ms: 4000, audio: null, load: 0.2 },
];

/**
 * Voz sintetica: silabas de ~180 ms con dos formantes que se mueven, para que
 * la esfera se mueva como con una voz y no como con ruido blanco.
 */
class Simulation {
  private index = 0;
  private current: ScriptStep | null = null;
  private stateTimer: number | null = null;
  private audioTimer: number | null = null;
  private previous: CoreState = "DORMIDO";
  private readonly startedAt = performance.now();

  constructor(private readonly emit: (message: CoreMessage) => void) {}

  start(): void {
    this.next();
    this.audioTimer = window.setInterval(() => this.tick(), 1000 / 60);
  }

  stop(): void {
    if (this.stateTimer !== null) window.clearTimeout(this.stateTimer);
    if (this.audioTimer !== null) window.clearInterval(this.audioTimer);
  }

  private next(): void {
    const step = SCRIPT[this.index % SCRIPT.length];
    if (step === undefined) return;
    this.current = step;
    this.emit({ type: "state", state: step.state, previous: this.previous, reason: "simulacion" });
    this.emit({ type: "cognition", load: step.load, level: step.state === "PENSANDO" ? 2 : 0 });
    this.previous = step.state;
    this.index += 1;
    this.stateTimer = window.setTimeout(() => this.next(), step.ms);
  }

  private tick(): void {
    const step = this.current;
    if (step === null || step.audio === null) return;
    const t = (performance.now() - this.startedAt) / 1000;
    const syllable = Math.max(0, Math.sin(t * Math.PI * 5.5)) ** 0.6;
    const phrase = Math.max(0, Math.sin(t * 0.9)) * 0.5 + 0.5;
    const level = syllable * phrase * (step.audio === "tts" ? 0.85 : 0.65);
    const formant1 = 4 + Math.sin(t * 3.1) * 2;
    const formant2 = 12 + Math.sin(t * 2.3) * 4;
    const bands: number[] = [];
    for (let band = 0; band < BAND_COUNT; band += 1) {
      const f1 = Math.exp(-((band - formant1) ** 2) / 6);
      const f2 = 0.6 * Math.exp(-((band - formant2) ** 2) / 10);
      bands.push(Math.min(1, (f1 + f2) * level * 1.3 + 0.08 * Math.random() * level));
    }
    this.emit({ type: "audio", source: step.audio, level, bands });
  }
}

export const coreLink = new CoreLink();
