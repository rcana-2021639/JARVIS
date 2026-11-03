/**
 * El audio que mueve la esfera, suavizado para que sea organico y no epileptico.
 *
 * Consume `audio` del nucleo (o de la simulacion) y expone, a 60 fps y sin
 * re-renderizar React:
 *
 *   level   amplitud general, 0..1
 *   bands   32 bandas de frecuencia, 0..1 (Float32Array, lista para el uniforme)
 *   bass    energia de graves (las 4 primeras bandas)
 *   beat    1 en el instante de un golpe, decae a 0 en ~200 ms
 *   source  de quien es la voz: "mic" (el usuario) o "tts" (Jarvis)
 *
 * DECISION: suavizado exponencial asimetrico. Sube rapido (ataque 0.5) para no
 * llegar tarde a una silaba, y baja lento (relajacion 0.12) para que la malla
 * no tiemble entre bloques. Es lo mismo que hace un vumetro analogico.
 */

import { useEffect, useRef } from "react";

import { coreLink } from "../services/core";
import { BAND_COUNT } from "../services/protocol";

export interface AudioFrame {
  level: number;
  readonly bands: Float32Array;
  bass: number;
  beat: number;
  source: "mic" | "tts" | null;
}

const ATTACK = 0.5;
const RELEASE = 0.12;
/** Sin bloques nuevos en este tiempo, el objetivo vuelve a silencio. */
const STALE_MS = 150;
const BEAT_RATIO = 1.45;
const BEAT_DECAY_PER_S = 5;

function smooth(current: number, target: number): number {
  const k = target > current ? ATTACK : RELEASE;
  return current + (target - current) * k;
}

/** Mantiene el ultimo bloque recibido y lo convierte en un cuadro suavizado. */
export class AudioReactive {
  readonly frame: AudioFrame = {
    level: 0,
    bands: new Float32Array(BAND_COUNT),
    bass: 0,
    beat: 0,
    source: null,
  };
  private readonly target = new Float32Array(BAND_COUNT);
  private targetLevel = 0;
  private lastAt = 0;
  private bassAverage = 0;

  push(level: number, bands: readonly number[], source: "mic" | "tts"): void {
    this.targetLevel = Math.min(1, Math.max(0, level));
    for (let i = 0; i < BAND_COUNT; i += 1) {
      this.target[i] = Math.min(1, Math.max(0, bands[i] ?? 0));
    }
    this.frame.source = source;
    this.lastAt = performance.now();
  }

  /** Avanza un cuadro. Se llama desde useFrame con el delta en segundos. */
  step(delta: number): AudioFrame {
    const stale = performance.now() - this.lastAt > STALE_MS;
    const frame = this.frame;
    frame.level = smooth(frame.level, stale ? 0 : this.targetLevel);
    let bass = 0;
    for (let i = 0; i < BAND_COUNT; i += 1) {
      const value = smooth(frame.bands[i] ?? 0, stale ? 0 : (this.target[i] ?? 0));
      frame.bands[i] = value;
      if (i < 4) bass += value / 4;
    }
    frame.bass = bass;
    // Golpe: los graves saltan por encima de su promedio reciente.
    this.bassAverage += (bass - this.bassAverage) * 0.05;
    if (bass > 0.08 && bass > this.bassAverage * BEAT_RATIO && frame.beat < 0.3) {
      frame.beat = 1;
    } else {
      frame.beat = Math.max(0, frame.beat - delta * BEAT_DECAY_PER_S);
    }
    if (stale && frame.level < 0.01) frame.source = null;
    return frame;
  }
}

/** Un `AudioReactive` conectado al nucleo durante la vida del componente. */
export function useAudioReactive(): AudioReactive {
  const reactive = useRef<AudioReactive>(new AudioReactive());
  useEffect(
    () =>
      coreLink.onMessage((message) => {
        if (message.type === "audio") {
          reactive.current.push(message.level, message.bands, message.source);
        }
      }),
    [],
  );
  return reactive.current;
}
