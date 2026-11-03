/**
 * Capa 5 — el campo de fresnel.
 *
 * Un caparazon translucido: transparente al centro, brillante en los bordes.
 * Es lo que hace que el conjunto se sienta como energia contenida y no como
 * una bola solida.
 */

import { useFrame } from "@react-three/fiber";
import { useMemo } from "react";
import { AdditiveBlending, Color, FrontSide } from "three";

import { useReactor } from "./ReactorContext";

const VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vView = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}
`;

const FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uIntensity;
uniform float uPower;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  float fresnel = pow(1.0 - max(dot(normalize(vNormal), normalize(vView)), 0.0), uPower);
  gl_FragColor = vec4(uColor * fresnel * uIntensity, fresnel);
}
`;

export function EnergyField(): JSX.Element {
  const { frame, blender, audio } = useReactor();
  const uniforms = useMemo(
    () => ({ uColor: { value: new Color() }, uIntensity: { value: 0.6 }, uPower: { value: 2.6 } }),
    [],
  );

  useFrame(() => {
    const p = blender.current;
    uniforms.uColor.value.copy(p.field);
    // CONFIRMANDO late como un aviso; HABLANDO respira con la voz.
    const wait = p.freeze * 0.35 * (0.5 + 0.5 * Math.sin(frame.time * Math.PI * 2 * 0.9));
    uniforms.uIntensity.value = 0.16 + 0.2 * p.shellIntensity + audio.frame.level * 0.35 + wait;
    uniforms.uPower.value = 3.8 - audio.frame.level * 1.0;
  });

  return (
    <mesh scale={1.3}>
      <sphereGeometry args={[1, 64, 64]} />
      <shaderMaterial
        vertexShader={VERTEX}
        fragmentShader={FRAGMENT}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
        side={FrontSide}
        toneMapped={false}
      />
    </mesh>
  );
}
