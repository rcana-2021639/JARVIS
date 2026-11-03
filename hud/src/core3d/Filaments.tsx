/**
 * Capa 4 — filamentos: curvas de Bezier entre puntos de la malla.
 *
 * Aparecen, se sostienen y se desvanecen. Cuantos hay vivos a la vez es
 * proporcional a la carga cognitiva: cuando Jarvis piensa fuerte, la esfera se
 * llena de conexiones.
 *
 * DECISION: todos los filamentos viven en UNA geometria de segmentos con color
 * por vertice (con mezcla aditiva, el brillo hace de opacidad). Son 48 curvas
 * y una sola llamada de dibujo, en vez de 48 objetos.
 */

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  type LineSegments,
  QuadraticBezierCurve3,
  Vector3,
} from "three";

import { useReactor } from "./ReactorContext";

const MAX_FILAMENTS = 48;
const SEGMENTS = 24;
const VERTS_PER = SEGMENTS * 2;

interface Filament {
  age: number;
  life: number;
  alive: boolean;
}

function randomOnSphere(target: Vector3): Vector3 {
  const u = Math.random() * 2 - 1;
  const theta = Math.random() * Math.PI * 2;
  const r = Math.sqrt(1 - u * u);
  return target.set(r * Math.cos(theta), u, r * Math.sin(theta));
}

/** Entra en 0.3 s, se sostiene, sale en 0.5 s. */
function envelope(f: Filament): number {
  const fadeIn = Math.min(1, f.age / 0.3);
  const fadeOut = Math.min(1, (f.life - f.age) / 0.5);
  return Math.max(0, Math.min(fadeIn, fadeOut));
}

export function Filaments(): JSX.Element {
  const { frame, blender } = useReactor();
  const lines = useRef<LineSegments>(null);
  const pool = useMemo<Filament[]>(
    () => Array.from({ length: MAX_FILAMENTS }, () => ({ age: 0, life: 0, alive: false })),
    [],
  );
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(new Float32Array(MAX_FILAMENTS * VERTS_PER * 3), 3));
    g.setAttribute("color", new BufferAttribute(new Float32Array(MAX_FILAMENTS * VERTS_PER * 3), 3));
    return g;
  }, []);
  const temp = useMemo(
    () => ({ a: new Vector3(), b: new Vector3(), c: new Vector3(), p: new Vector3() }),
    [],
  );

  function spawn(index: number): void {
    const f = pool[index];
    if (f === undefined) return;
    const { a, b, c, p } = temp;
    randomOnSphere(a);
    // El otro extremo, cerca: arcos cortos se leen como conexiones, no como ruido.
    randomOnSphere(b).multiplyScalar(0.7).add(a).normalize();
    c.copy(a).add(b).normalize().multiplyScalar(1.18 + Math.random() * 0.25);
    const curve = new QuadraticBezierCurve3(a.clone(), c.clone(), b.clone());
    const positions = geometry.getAttribute("position") as BufferAttribute;
    const base = index * VERTS_PER;
    for (let s = 0; s < SEGMENTS; s += 1) {
      curve.getPoint(s / SEGMENTS, p);
      positions.setXYZ(base + s * 2, p.x, p.y, p.z);
      curve.getPoint((s + 1) / SEGMENTS, p);
      positions.setXYZ(base + s * 2 + 1, p.x, p.y, p.z);
    }
    positions.needsUpdate = true;
    f.age = 0;
    f.life = 0.8 + Math.random() * 1.6;
    f.alive = true;
  }

  useFrame((_, delta) => {
    const p = blender.current;
    const target = Math.round(MAX_FILAMENTS * Math.min(1, frame.load * p.filaments + p.filaments * 0.08));
    let alive = 0;
    for (const f of pool) {
      if (!f.alive) continue;
      f.age += delta * (1 - p.freeze * 0.9);
      if (f.age >= f.life) f.alive = false;
      else alive += 1;
    }
    for (let i = 0; i < pool.length && alive < target; i += 1) {
      if (!pool[i]?.alive && Math.random() < 0.25) {
        spawn(i);
        alive += 1;
      }
    }
    const colors = geometry.getAttribute("color") as BufferAttribute;
    pool.forEach((f, index) => {
      const k = f.alive ? envelope(f) * 0.85 : 0;
      for (let v = 0; v < VERTS_PER; v += 1) {
        // Mas brillo en el centro del arco: la energia viaja.
        const along = Math.sin((Math.floor(v / 2) / SEGMENTS) * Math.PI);
        const glow = k * (0.35 + 0.65 * along);
        colors.setXYZ(index * VERTS_PER + v, p.rings.r * glow, p.rings.g * glow, p.rings.b * glow);
      }
    });
    colors.needsUpdate = true;
    lines.current?.scale.setScalar(p.expansion);
  });

  return (
    <lineSegments ref={lines} geometry={geometry} frustumCulled={false}>
      <lineBasicMaterial
        vertexColors
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
        toneMapped={false}
      />
    </lineSegments>
  );
}
