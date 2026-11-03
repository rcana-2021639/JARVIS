/**
 * El nucleo completo: las seis capas, su reloj comun y el nivel de detalle.
 *
 *   Seed           capa 1, el corazon
 *   ParticleShell  capa 2, la malla reactiva al audio
 *   Rings          capa 3, anillos orbitales
 *   Filaments      capa 4, carga cognitiva
 *   EnergyField    capa 5, fresnel
 *   Effects        capa 6, postproceso
 *
 * RENDIMIENTO: si los fotogramas caen, primero se bajan particulas y la
 * densidad de pixeles, despues pasadas de bloom; nunca la fluidez. Con la
 * ventana oculta no se dibuja nada.
 */

import { PerformanceMonitor } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { type MutableRefObject, useEffect, useMemo, useRef, useState } from "react";
import type { Group } from "three";

import { color } from "../design/tokens";
import { useAudioReactive } from "../hooks/useAudioReactive";
import { coreLink } from "../services/core";
import type { CoreState } from "../services/protocol";
import { Effects } from "./Effects";
import { EnergyField } from "./EnergyField";
import { Filaments } from "./Filaments";
import { ParticleShell } from "./ParticleShell";
import { ReactorContext, type ReactorFrame, type ReactorShared, useReactor } from "./ReactorContext";
import { Rings } from "./Rings";
import { Seed } from "./Seed";
import { StateBlender } from "./states";

/** Niveles de detalle: particulas por nivel. El documento pide 30.000 en el primero. */
export const PARTICLE_LOD = [30000, 20000, 12000, 7000] as const;
const MAX_DPR = 1.5;
const ERROR_SHOCK_S = 0.9;

declare global {
  interface Window {
    /** Fotogramas por segundo, uno por segundo: el banco de 5 minutos los lee de aqui. */
    __jarvisFps?: number[];
  }
}

interface DriverProps {
  readonly target: MutableRefObject<CoreState>;
  readonly loadTarget: MutableRefObject<number>;
}

/** Avanza el reloj, el audio y los estados una vez por cuadro, antes que las capas. */
function Driver({ target, loadTarget }: DriverProps): null {
  const { frame, blender, audio } = useReactor();
  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const dt = Math.min(delta, 0.1);
    blender.setTarget(target.current, t);
    const p = blender.sample(t);
    audio.step(dt);
    frame.time = t;
    frame.motionTime += dt * (1 - p.freeze * 0.92);
    frame.load += (loadTarget.current - frame.load) * Math.min(1, dt * 2.5);
    // ERROR: sacudida y fragmentacion breve, despues se recompone sola.
    frame.shock =
      blender.state === "ERROR"
        ? Math.max(0, 1 - blender.sinceChange / ERROR_SHOCK_S)
        : Math.max(0, frame.shock - dt * 3);
  }, -1);
  return null;
}

function Spin({ children }: { readonly children: JSX.Element[] }): JSX.Element {
  const { blender } = useReactor();
  const group = useRef<Group>(null);
  useFrame((_, delta) => {
    const g = group.current;
    if (g === null) return;
    g.rotation.y += delta * 0.06 * (1 - blender.current.freeze);
    g.rotation.x = 0.18;
  });
  return <group ref={group}>{children}</group>;
}

function FpsProbe({ onFps }: { readonly onFps: (fps: number) => void }): null {
  const counter = useRef({ frames: 0, since: performance.now() });
  useFrame(() => {
    const c = counter.current;
    c.frames += 1;
    const now = performance.now();
    if (now - c.since >= 1000) {
      const fps = (c.frames * 1000) / (now - c.since);
      c.frames = 0;
      c.since = now;
      const history = (window.__jarvisFps ??= []);
      history.push(Math.round(fps * 10) / 10);
      if (history.length > 900) history.shift();
      onFps(fps);
    }
  });
  return null;
}

export interface ReactorProps {
  readonly onFps?: (fps: number) => void;
  readonly onLod?: (particles: number) => void;
}

export function Reactor({ onFps, onLod }: ReactorProps): JSX.Element {
  const audio = useAudioReactive();
  const shared = useMemo<ReactorShared>(() => {
    const frame: ReactorFrame = { time: 0, motionTime: 0, load: 0, shock: 0 };
    return { frame, blender: new StateBlender(), audio };
  }, [audio]);
  const target = useRef<CoreState>(coreLink.current().state);
  const loadTarget = useRef(0);
  const [lod, setLod] = useState(0);
  const [dpr, setDpr] = useState(Math.min(window.devicePixelRatio || 1, MAX_DPR));
  const [visible, setVisible] = useState(!document.hidden);

  useEffect(() => coreLink.onStatus((snapshot) => (target.current = snapshot.state)), []);
  useEffect(
    () =>
      coreLink.onMessage((message) => {
        if (message.type === "cognition") loadTarget.current = Math.min(1, Math.max(0, message.load));
      }),
    [],
  );
  useEffect(() => {
    const update = (): void => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);
  const particles = PARTICLE_LOD[lod] ?? PARTICLE_LOD[0];
  useEffect(() => onLod?.(particles), [particles, onLod]);

  return (
    <Canvas
      frameloop={visible ? "always" : "never"}
      dpr={dpr}
      gl={{ antialias: false, powerPreference: "high-performance", alpha: false, stencil: false }}
      camera={{ position: [0, 0, 5.2], fov: 40, near: 0.1, far: 50 }}
    >
      <color attach="background" args={[color.void]} />
      <PerformanceMonitor
        flipflops={4}
        onDecline={() => {
          setLod((value) => Math.min(value + 1, PARTICLE_LOD.length - 1));
          setDpr((value) => Math.max(1, value - 0.25));
        }}
        onIncline={() => {
          setLod((value) => Math.max(value - 1, 0));
          setDpr((value) => Math.min(MAX_DPR, value + 0.25));
        }}
        onFallback={() => setLod(PARTICLE_LOD.length - 1)}
      >
        <ReactorContext.Provider value={shared}>
          <Driver target={target} loadTarget={loadTarget} />
          <Spin>
            <Seed />
            <ParticleShell count={particles} />
            <Filaments />
            <EnergyField />
            <Rings />
          </Spin>
          <Effects lite={lod >= 2} />
          <FpsProbe onFps={(fps) => onFps?.(fps)} />
        </ReactorContext.Provider>
      </PerformanceMonitor>
    </Canvas>
  );
}
