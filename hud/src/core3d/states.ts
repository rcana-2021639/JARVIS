/**
 * La maquina de estados visual: los ocho estados del nucleo y sus transiciones.
 *
 * Cada estado es un juego de parametros (colores, expansion, ruido, velocidad
 * de anillos, filamentos...). Al cambiar de estado no se salta: se interpola
 * del punto donde iba la animacion al nuevo estado en 800 ms con la curva
 * estandar. Si llega otro cambio a mitad de camino, parte desde donde esta.
 */

import { Color } from "three";

import { color, ease, motion } from "../design/tokens";
import type { CoreState } from "../services/protocol";

export interface VisualParams {
  /** Color del corazon (semilla) y su intensidad emisiva. */
  seed: Color;
  seedIntensity: number;
  /** Amplitud de la respiracion de 4 s (0 = quieto). */
  breath: number;
  /** Colores de la malla: base y acento segun la velocidad radial. */
  shellA: Color;
  shellB: Color;
  shellIntensity: number;
  /** Color del campo de fresnel y de los anillos. */
  field: Color;
  rings: Color;
  /** Radio relativo de la esfera; < 1 colapsa hacia el centro. */
  expansion: number;
  noiseAmp: number;
  noiseSpeed: number;
  /** Cuanto deforma el audio a la malla. */
  audioGain: number;
  /** Giro en espiral de las particulas (PENSANDO). */
  spiral: number;
  ringSpeed: number;
  /** 0 = cada anillo en su eje; 1 = todos alineados en un plano (EJECUTANDO). */
  ringAlign: number;
  /** El anillo exterior de escucha / espera. */
  outerRing: number;
  /** Barra de progreso orbital (EJECUTANDO). */
  progress: number;
  /** Multiplicador de la densidad de filamentos sobre la carga cognitiva. */
  filaments: number;
  /** Pulsos que salen del centro (HABLANDO) o latido de espera (CONFIRMANDO). */
  pulse: number;
  /** 1 = todo detenido (CONFIRMANDO). */
  freeze: number;
}

function params(values: {
  seed: string;
  seedIntensity: number;
  breath: number;
  shellA: string;
  shellB: string;
  shellIntensity: number;
  field: string;
  rings: string;
  expansion: number;
  noiseAmp: number;
  noiseSpeed: number;
  audioGain: number;
  spiral: number;
  ringSpeed: number;
  ringAlign: number;
  outerRing: number;
  progress: number;
  filaments: number;
  pulse: number;
  freeze: number;
}): VisualParams {
  return {
    ...values,
    seed: new Color(values.seed),
    shellA: new Color(values.shellA),
    shellB: new Color(values.shellB),
    field: new Color(values.field),
    rings: new Color(values.rings),
  };
}

const BASE = {
  breath: 0,
  spiral: 0,
  ringAlign: 0,
  outerRing: 0,
  progress: 0,
  filaments: 0.3,
  pulse: 0,
  freeze: 0,
};

export const STATE_PARAMS: Record<CoreState, VisualParams> = {
  DORMIDO: params({
    ...BASE,
    seed: color.amberCore, seedIntensity: 1.1, breath: 0.35,
    shellA: color.amberCore, shellB: color.amberHot, shellIntensity: 0.35,
    field: color.amberCore, rings: color.cyanDim,
    expansion: 0.9, noiseAmp: 0.035, noiseSpeed: 0.12, audioGain: 0.2, ringSpeed: 0.12,
    filaments: 0.05,
  }),
  DESPIERTO: params({
    ...BASE,
    seed: color.cyanGlow, seedIntensity: 2.4,
    shellA: color.cyanCore, shellB: color.cyanGlow, shellIntensity: 0.8,
    field: color.cyanCore, rings: color.cyanCore,
    expansion: 1.14, noiseAmp: 0.08, noiseSpeed: 0.5, audioGain: 0.8, ringSpeed: 1.6,
    outerRing: 1,
  }),
  ESCUCHANDO: params({
    ...BASE,
    seed: color.cyanCore, seedIntensity: 1.8,
    shellA: color.cyanCore, shellB: color.amberHot, shellIntensity: 0.75,
    field: color.cyanCore, rings: color.cyanCore,
    expansion: 1.04, noiseAmp: 0.05, noiseSpeed: 0.35, audioGain: 1.1, ringSpeed: 0.8,
    outerRing: 1,
  }),
  PENSANDO: params({
    ...BASE,
    seed: color.cyanGlow, seedIntensity: 2.0,
    shellA: color.violet, shellB: color.cyanGlow, shellIntensity: 0.8,
    field: color.violet, rings: color.cyanGlow,
    expansion: 0.72, noiseAmp: 0.09, noiseSpeed: 0.9, audioGain: 0.3, ringSpeed: 2.8,
    spiral: 1, filaments: 1,
  }),
  HABLANDO: params({
    ...BASE,
    seed: color.amberHot, seedIntensity: 2.2,
    shellA: color.amberCore, shellB: color.amberHot, shellIntensity: 0.6,
    field: color.amberCore, rings: color.amberCore,
    expansion: 1.0, noiseAmp: 0.045, noiseSpeed: 0.3, audioGain: 1.35, ringSpeed: 0.7,
    pulse: 1, filaments: 0.4,
  }),
  EJECUTANDO: params({
    ...BASE,
    seed: color.mint, seedIntensity: 1.9,
    shellA: color.mint, shellB: color.cyanGlow, shellIntensity: 0.5,
    field: color.mint, rings: color.mint,
    expansion: 0.98, noiseAmp: 0.04, noiseSpeed: 0.4, audioGain: 0.4, ringSpeed: 1.3,
    ringAlign: 1, progress: 1, filaments: 0.5,
  }),
  ERROR: params({
    ...BASE,
    seed: color.crimson, seedIntensity: 2.0,
    shellA: color.crimson, shellB: color.amberHot, shellIntensity: 0.75,
    field: color.crimson, rings: color.crimson,
    expansion: 1.0, noiseAmp: 0.16, noiseSpeed: 1.4, audioGain: 0.3, ringSpeed: 0.5,
  }),
  CONFIRMANDO: params({
    ...BASE,
    seed: color.crimson, seedIntensity: 1.6,
    shellA: color.crimson, shellB: color.amberCore, shellIntensity: 0.8,
    field: color.crimson, rings: color.crimson,
    expansion: 0.96, noiseAmp: 0.02, noiseSpeed: 0.05, audioGain: 0.2, ringSpeed: 0,
    outerRing: 1, pulse: 1, freeze: 1, filaments: 0,
  }),
};

const NUMERIC_KEYS = [
  "seedIntensity", "breath", "shellIntensity", "expansion", "noiseAmp", "noiseSpeed",
  "audioGain", "spiral", "ringSpeed", "ringAlign", "outerRing", "progress", "filaments",
  "pulse", "freeze",
] as const;
const COLOR_KEYS = ["seed", "shellA", "shellB", "field", "rings"] as const;

function copyParams(source: VisualParams, target: VisualParams): void {
  for (const key of NUMERIC_KEYS) target[key] = source[key];
  for (const key of COLOR_KEYS) target[key].copy(source[key]);
}

function cloneParams(source: VisualParams): VisualParams {
  const clone = { ...source };
  for (const key of COLOR_KEYS) clone[key] = source[key].clone();
  return clone;
}

/** Interpola entre estados sin asignar memoria por cuadro. */
export class StateBlender {
  state: CoreState = "DORMIDO";
  /** Segundos desde que se entro al estado actual (para impulsos como el del error). */
  sinceChange = 0;
  readonly current: VisualParams = cloneParams(STATE_PARAMS.DORMIDO);
  private readonly from: VisualParams = cloneParams(STATE_PARAMS.DORMIDO);
  private startedAt = 0;
  private transitioning = false;

  setTarget(state: CoreState, now: number): void {
    if (state === this.state) return;
    copyParams(this.current, this.from);
    this.state = state;
    this.startedAt = now;
    this.sinceChange = 0;
    this.transitioning = true;
  }

  /** Avanza a `now` (segundos) y devuelve los parametros interpolados. */
  sample(now: number): VisualParams {
    this.sinceChange = now - this.startedAt;
    if (!this.transitioning) return this.current;
    const t = Math.min(1, (now - this.startedAt) / (motion.cinematicMs / 1000));
    const k = ease(t);
    const target = STATE_PARAMS[this.state];
    for (const key of NUMERIC_KEYS) {
      this.current[key] = this.from[key] + (target[key] - this.from[key]) * k;
    }
    for (const key of COLOR_KEYS) {
      this.current[key].copy(this.from[key]).lerp(target[key], k);
    }
    if (t >= 1) this.transitioning = false;
    return this.current;
  }
}
