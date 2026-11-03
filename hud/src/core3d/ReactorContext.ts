/**
 * Lo que comparten las seis capas de la esfera en cada cuadro.
 *
 * Un solo `useFrame` (en `Reactor`) avanza el reloj, el audio y la maquina de
 * estados; las capas leen de aqui en sus propios `useFrame`. Nada de esto pasa
 * por el estado de React: re-renderizar a 60 fps seria tirar el presupuesto.
 */

import { createContext, useContext } from "react";

import type { AudioReactive } from "../hooks/useAudioReactive";
import type { StateBlender } from "./states";

export interface ReactorFrame {
  /** Segundos desde que arranco la escena. */
  time: number;
  /** Tiempo "congelable": en CONFIRMANDO avanza casi nada. */
  motionTime: number;
  /** Carga cognitiva suavizada, 0..1. */
  load: number;
  /** Impulso del error: 1 al entrar en ERROR, decae a 0. */
  shock: number;
}

export interface ReactorShared {
  readonly frame: ReactorFrame;
  readonly blender: StateBlender;
  readonly audio: AudioReactive;
}

export const ReactorContext = createContext<ReactorShared | null>(null);

export function useReactor(): ReactorShared {
  const shared = useContext(ReactorContext);
  if (shared === null) throw new Error("useReactor fuera de <Reactor>");
  return shared;
}
