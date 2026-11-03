/**
 * Capa 1 — la semilla: el corazon de energia.
 *
 * Una esfera pequena de material emisivo que palpita lento (4 s). Su
 * intensidad es la linea base de energia del sistema; el bloom la convierte
 * en el resplandor del reactor.
 */

import { Billboard } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import { AdditiveBlending, Color, type Mesh, type MeshBasicMaterial } from "three";

import { motion } from "../design/tokens";
import { useReactor } from "./ReactorContext";

/** La corona: un disco siempre de frente con brillo que cae como 1/r. */
const CORONA_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CORONA_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uEnergy;
varying vec2 vUv;
void main() {
  float r = length(vUv - 0.5) * 2.0;
  float glow = pow(max(0.0, 1.0 - r), 2.2) * 0.9 + 0.06 / (r + 0.08) * (1.0 - r);
  gl_FragColor = vec4(uColor * glow * uEnergy, glow);
}
`;

export function Seed(): JSX.Element {
  const { frame, blender, audio } = useReactor();
  const mesh = useRef<Mesh>(null);
  const material = useRef<MeshBasicMaterial>(null);
  const corona = useMemo(() => ({ uColor: { value: new Color() }, uEnergy: { value: 1 } }), []);

  useFrame(() => {
    const p = blender.current;
    const heartbeat = 0.5 + 0.5 * Math.sin((frame.time * Math.PI * 2 * 1000) / motion.breathMs);
    const energy = p.seedIntensity * (0.75 + 0.25 * heartbeat) + audio.frame.level * 1.2;
    material.current?.color.copy(p.seed).multiplyScalar(energy * 1.6);
    corona.uColor.value.copy(p.seed);
    corona.uEnergy.value = 0.55 * energy;
    const scale = p.expansion * (0.95 + 0.08 * heartbeat + audio.frame.beat * 0.12);
    mesh.current?.scale.setScalar(scale);
  });

  return (
    <group>
      <mesh ref={mesh}>
        <sphereGeometry args={[0.15, 48, 48]} />
        <meshBasicMaterial ref={material} toneMapped={false} />
      </mesh>
      <Billboard>
        <mesh>
          <planeGeometry args={[1.5, 1.5]} />
          <shaderMaterial
            vertexShader={CORONA_VERTEX}
            fragmentShader={CORONA_FRAGMENT}
            uniforms={corona}
            transparent
            depthWrite={false}
            blending={AdditiveBlending}
            toneMapped={false}
          />
        </mesh>
      </Billboard>
    </group>
  );
}
