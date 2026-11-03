/**
 * Lo que el nucleo Python manda al HUD por el WebSocket local.
 *
 * Es una vista compacta del bus (ver `core/jarvis/hudlink/feed.py`): el HUD no
 * recibe cada evento crudo, solo lo que necesita para dibujar a 60 fps.
 */

export const CORE_STATES = [
  "DORMIDO",
  "DESPIERTO",
  "ESCUCHANDO",
  "PENSANDO",
  "HABLANDO",
  "EJECUTANDO",
  "ERROR",
  "CONFIRMANDO",
] as const;

export type CoreState = (typeof CORE_STATES)[number];

export const BAND_COUNT = 32;

export interface HelloMessage {
  readonly type: "hello";
  readonly version: number;
  readonly bands: number;
  readonly state: CoreState;
}

export interface StateMessage {
  readonly type: "state";
  readonly state: CoreState;
  readonly previous: CoreState;
  readonly reason: string;
}

export interface AudioMessage {
  readonly type: "audio";
  /** "mic": la voz del usuario entrando; "tts": la de Jarvis saliendo. */
  readonly source: "mic" | "tts";
  readonly level: number;
  readonly bands: readonly number[];
}

export interface CognitionMessage {
  readonly type: "cognition";
  /** Carga cognitiva 0..1: alimenta la densidad de filamentos. */
  readonly load: number;
  readonly level: number;
}

export type CoreMessage = HelloMessage | StateMessage | AudioMessage | CognitionMessage;

function isState(value: unknown): value is CoreState {
  return typeof value === "string" && (CORE_STATES as readonly string[]).includes(value);
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number");
}

/** Valida un mensaje recibido. Lo que no se reconoce se descarta: nunca rompe el HUD. */
export function parseMessage(raw: string): CoreMessage | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null) return null;
  const msg = data as Record<string, unknown>;
  switch (msg["type"]) {
    case "hello": {
      const state = msg["state"];
      const version = msg["version"];
      if (!isState(state) || typeof version !== "number") return null;
      return { type: "hello", version, bands: Number(msg["bands"] ?? BAND_COUNT), state };
    }
    case "state": {
      const state = msg["state"];
      const previous = msg["previous"];
      if (!isState(state) || !isState(previous)) return null;
      return { type: "state", state, previous, reason: String(msg["reason"] ?? "") };
    }
    case "audio": {
      const source = msg["source"];
      const level = msg["level"];
      const bands = msg["bands"];
      if ((source !== "mic" && source !== "tts") || typeof level !== "number") return null;
      if (!isNumberArray(bands)) return null;
      return { type: "audio", source, level, bands };
    }
    case "cognition": {
      const load = msg["load"];
      if (typeof load !== "number") return null;
      return { type: "cognition", load, level: Number(msg["level"] ?? 0) };
    }
    default:
      return null;
  }
}
