/**
 * Capa 3 — anillos orbitales con graduacion de instrumento.
 *
 * Cuatro anillos en ejes distintos, cada uno con marcas cada 5 grados (largas
 * cada 30). Giran a velocidades NO armonicas entre si —razones irracionales—
 * para que nunca se sincronicen. La velocidad escala con el estado; en
 * EJECUTANDO se alinean en un plano y aparece la barra de progreso orbital.
 * El anillo exterior de escucha aparece al despertar y en la espera de una
 * confirmacion.
 */

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BufferGeometry,
  Euler,
  Float32BufferAttribute,
  type Group,
  type LineBasicMaterial,
  Quaternion,
} from "three";

import { useReactor } from "./ReactorContext";

const SQRT2 = Math.SQRT2;
const PHI = (1 + Math.sqrt(5)) / 2;

interface RingSpec {
  readonly radius: number;
  readonly tilt: Euler;
  /** Radianes por segundo a velocidad 1. Razones irracionales entre ellos. */
  readonly speed: number;
  readonly ticks: boolean;
}

const RINGS: readonly RingSpec[] = [
  { radius: 1.42, tilt: new Euler(1.22, 0.18, 0), speed: 0.21, ticks: true },
  { radius: 1.56, tilt: new Euler(0.38, 1.05, 0.31), speed: -0.21 / PHI, ticks: true },
  { radius: 1.68, tilt: new Euler(-0.92, 0.47, 0.83), speed: 0.21 / SQRT2, ticks: false },
  { radius: 1.82, tilt: new Euler(0.14, -0.71, -1.13), speed: -0.21 / Math.E, ticks: true },
];

/** El plano comun en el que se alinean en EJECUTANDO: inclinado como un anillo planetario. */
const ALIGNED = new Quaternion().setFromEuler(new Euler(1.25, 0, 0.22));

function ringGeometry(radius: number, ticks: boolean, arc = Math.PI * 2): BufferGeometry {
  const points: number[] = [];
  const segments = 192;
  for (let i = 0; i < segments; i += 1) {
    const a0 = (i / segments) * arc;
    const a1 = ((i + 1) / segments) * arc;
    points.push(Math.cos(a0) * radius, Math.sin(a0) * radius, 0);
    points.push(Math.cos(a1) * radius, Math.sin(a1) * radius, 0);
  }
  if (ticks) {
    for (let deg = 0; deg < 360; deg += 5) {
      const a = (deg * Math.PI) / 180;
      const length = deg % 30 === 0 ? 0.075 : 0.03;
      const inner = radius - length;
      points.push(Math.cos(a) * inner, Math.sin(a) * inner, 0);
      points.push(Math.cos(a) * radius, Math.sin(a) * radius, 0);
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(points, 3));
  return geometry;
}

function dashedRing(radius: number, dashes: number): BufferGeometry {
  const points: number[] = [];
  for (let i = 0; i < dashes; i += 1) {
    const a0 = (i / dashes) * Math.PI * 2;
    const a1 = a0 + (Math.PI * 2) / dashes / 2;
    for (let k = 0; k < 6; k += 1) {
      const t0 = a0 + ((a1 - a0) * k) / 6;
      const t1 = a0 + ((a1 - a0) * (k + 1)) / 6;
      points.push(Math.cos(t0) * radius, Math.sin(t0) * radius, 0);
      points.push(Math.cos(t1) * radius, Math.sin(t1) * radius, 0);
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(points, 3));
  return geometry;
}

const lineProps = {
  transparent: true,
  depthWrite: false,
  blending: AdditiveBlending,
  toneMapped: false,
} as const;

export function Rings(): JSX.Element {
  const { frame, blender, audio } = useReactor();
  const rings = useMemo(
    () =>
      RINGS.map((spec) => ({
        spec,
        geometry: ringGeometry(spec.radius, spec.ticks),
        tilt: new Quaternion().setFromEuler(spec.tilt),
      })),
    [],
  );
  const outerGeometry = useMemo(() => dashedRing(2.08, 48), []);
  const progressGeometry = useMemo(() => ringGeometry(1.96, false, Math.PI * 0.55), []);
  const holders = useRef<(Group | null)[]>([]);
  const spinners = useRef<(Group | null)[]>([]);
  const materials = useRef<(LineBasicMaterial | null)[]>([]);
  const outer = useRef<Group>(null);
  const outerMaterial = useRef<LineBasicMaterial>(null);
  const progress = useRef<Group>(null);
  const progressMaterial = useRef<LineBasicMaterial>(null);
  const angles = useRef<number[]>(RINGS.map(() => Math.random() * Math.PI * 2));
  const scratch = useMemo(() => new Quaternion(), []);

  useFrame((_, delta) => {
    const p = blender.current;
    const speed = p.ringSpeed * (1 - p.freeze);
    rings.forEach(({ spec, tilt }, index) => {
      const holder = holders.current[index];
      const spinner = spinners.current[index];
      const material = materials.current[index];
      if (!holder || !spinner || !material) return;
      const angle = (angles.current[index] ?? 0) + spec.speed * speed * delta * 6;
      angles.current[index] = angle;
      spinner.rotation.z = angle;
      holder.quaternion.copy(scratch.copy(tilt).slerp(ALIGNED, p.ringAlign));
      holder.scale.setScalar(p.expansion * 0.95 + 0.05 + audio.frame.beat * 0.015);
      material.color.copy(p.rings);
      material.opacity = 0.32 + 0.25 * audio.frame.level + 0.15 * (index % 2);
    });
    if (outer.current && outerMaterial.current) {
      const wait = p.freeze > 0.5 ? 0.55 + 0.45 * Math.sin(frame.time * Math.PI * 2 * 0.9) : 1;
      outerMaterial.current.opacity = p.outerRing * 0.55 * wait;
      outerMaterial.current.color.copy(p.rings);
      outer.current.rotation.z -= delta * 0.25 * (1 - p.freeze);
      outer.current.scale.setScalar(0.92 + p.outerRing * 0.08);
    }
    if (progress.current && progressMaterial.current) {
      progressMaterial.current.opacity = p.progress * 0.9;
      progressMaterial.current.color.copy(p.rings);
      progress.current.quaternion.copy(ALIGNED);
      progress.current.rotateZ(-frame.time * 2.2);
    }
  });

  return (
    <group>
      {rings.map(({ spec, geometry }, index) => (
        <group key={spec.radius} ref={(node) => (holders.current[index] = node)}>
          <group ref={(node) => (spinners.current[index] = node)}>
            <lineSegments geometry={geometry}>
              <lineBasicMaterial
                ref={(node) => (materials.current[index] = node)}
                {...lineProps}
                opacity={0.4}
              />
            </lineSegments>
          </group>
        </group>
      ))}
      <group ref={outer}>
        <lineSegments geometry={outerGeometry}>
          <lineBasicMaterial ref={outerMaterial} {...lineProps} opacity={0} />
        </lineSegments>
      </group>
      <group ref={progress}>
        <lineSegments geometry={progressGeometry}>
          <lineBasicMaterial ref={progressMaterial} {...lineProps} opacity={0} />
        </lineSegments>
      </group>
    </group>
  );
}
