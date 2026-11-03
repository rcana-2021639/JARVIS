/**
 * Capa 2 — la malla de particulas.
 *
 * N puntos en distribucion de Fibonacci sobre la esfera, desplazados en el
 * shader de vertices por ruido simplex 3D animado. CRITICO: el uniforme
 * `uAudio` trae las 32 bandas del analisis de frecuencia y cada particula se
 * desplaza segun la banda que le toca por su latitud (graves en el ecuador,
 * agudos en los polos): cuando Jarvis habla, la esfera toma la forma de su voz.
 * El color va de la base al acento segun la velocidad radial de la particula.
 */

import { useFrame } from "@react-three/fiber";
import { useMemo } from "react";
import { AdditiveBlending, BufferAttribute, BufferGeometry, Color } from "three";

import { BAND_COUNT } from "../services/protocol";
import { useReactor } from "./ReactorContext";
import { SIMPLEX_3D } from "./shaders/noise";

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

function fibonacciSphere(count: number): BufferGeometry {
  const positions = new Float32Array(count * 3);
  const bands = new Float32Array(count);
  const seeds = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / (count - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = i * GOLDEN_ANGLE;
    positions[i * 3] = Math.cos(theta) * radius;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = Math.sin(theta) * radius;
    // Banda fraccionaria: el shader interpola entre vecinas y no quedan escalones.
    bands[i] = Math.abs(y) * (BAND_COUNT - 1);
    seeds[i] = Math.random();
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  geometry.setAttribute("aBand", new BufferAttribute(bands, 1));
  geometry.setAttribute("aSeed", new BufferAttribute(seeds, 1));
  return geometry;
}

const VERTEX = /* glsl */ `
${SIMPLEX_3D}
uniform float uTime;
uniform float uAudio[${BAND_COUNT}];
uniform float uLevel;
uniform float uExpansion;
uniform float uNoiseAmp;
uniform float uNoiseSpeed;
uniform float uAudioGain;
uniform float uSpiral;
uniform float uShock;
uniform float uPulse;
uniform float uSize;
uniform float uPixelRatio;
attribute float aBand;
attribute float aSeed;
varying float vRadial;
varying float vGlow;

float displacement(vec3 dir, float t, float audio) {
  float n1 = snoise(dir * 1.6 + vec3(0.0, t * uNoiseSpeed, t * uNoiseSpeed * 0.3));
  float n2 = snoise(dir * 4.1 - vec3(t * uNoiseSpeed * 0.7));
  return n1 * uNoiseAmp + n2 * uNoiseAmp * 0.35 + audio * uAudioGain * 0.34 + uLevel * uAudioGain * 0.05;
}

void main() {
  vec3 dir = normalize(position);
  // PENSANDO: las particulas giran en espiral, mas rapido cerca del ecuador.
  float swirl = uSpiral * (uTime * 1.1 + dir.y * 3.0) * (1.0 - abs(dir.y));
  float c = cos(swirl);
  float s = sin(swirl);
  dir = vec3(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);

  int lowBand = int(floor(aBand));
  int highBand = min(lowBand + 1, ${BAND_COUNT - 1});
  float audio = mix(uAudio[lowBand], uAudio[highBand], fract(aBand));
  // El ruido rompe la simetria por latitud: la voz deforma, pero organica.
  audio *= 0.55 + 0.45 * snoise(dir * 2.2 + vec3(uTime * 0.4));
  float d = displacement(dir, uTime, audio);
  float before = displacement(dir, uTime - 0.06, audio);
  vRadial = (d - before) / 0.06;

  // HABLANDO: pulsos que salen del centro, como ondas.
  float wave = uPulse * uLevel * 0.08 * max(0.0, sin(uTime * 9.0 - aSeed * 1.5));

  float radius = uExpansion * (1.0 + d + wave);
  // ERROR: fragmentacion parcial (un tercio de las particulas sale disparado) y sacudida.
  radius += uShock * step(0.66, aSeed) * aSeed * 0.45;
  vec3 pos = dir * radius;
  pos += uShock * 0.035 * vec3(sin(uTime * 61.0 + aSeed * 9.0), cos(uTime * 53.0 + aSeed * 5.0), sin(uTime * 47.0 + aSeed));

  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uSize * uPixelRatio * (0.55 + aSeed * 0.9) * (1.0 + audio * 0.5) / -mv.z;
  vGlow = 0.45 + 0.55 * aSeed + audio * 0.3;
}
`;

const FRAGMENT = /* glsl */ `
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform float uIntensity;
varying float vRadial;
varying float vGlow;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  float core = smoothstep(0.5, 0.0, r);
  float accent = clamp(abs(vRadial) * 3.0, 0.0, 1.0);
  vec3 col = mix(uColorA, uColorB, accent);
  gl_FragColor = vec4(col * uIntensity * vGlow * core * 1.25, core);
}
`;

export interface ParticleShellProps {
  readonly count: number;
}

export function ParticleShell({ count }: ParticleShellProps): JSX.Element {
  const { frame, blender, audio } = useReactor();
  const geometry = useMemo(() => fibonacciSphere(count), [count]);
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uAudio: { value: new Float32Array(BAND_COUNT) },
      uLevel: { value: 0 },
      uExpansion: { value: 1 },
      uNoiseAmp: { value: 0.04 },
      uNoiseSpeed: { value: 0.2 },
      uAudioGain: { value: 0.2 },
      uSpiral: { value: 0 },
      uShock: { value: 0 },
      uPulse: { value: 0 },
      uSize: { value: 12 },
      uPixelRatio: { value: 1 },
      uColorA: { value: new Color() },
      uColorB: { value: new Color() },
      uIntensity: { value: 1 },
    }),
    [],
  );

  useFrame((state) => {
    const p = blender.current;
    const u = uniforms;
    u.uTime.value = frame.motionTime;
    u.uAudio.value.set(audio.frame.bands);
    u.uLevel.value = audio.frame.level;
    const breath = 1 + p.breath * 0.035 * Math.sin((frame.time * Math.PI * 2) / 4);
    u.uExpansion.value = p.expansion * breath + audio.frame.beat * 0.025;
    u.uNoiseAmp.value = p.noiseAmp;
    u.uNoiseSpeed.value = p.noiseSpeed;
    u.uAudioGain.value = p.audioGain;
    u.uSpiral.value = p.spiral;
    u.uShock.value = frame.shock;
    u.uPulse.value = p.pulse;
    u.uPixelRatio.value = state.viewport.dpr;
    u.uColorA.value.copy(p.shellA);
    u.uColorB.value.copy(p.shellB);
    u.uIntensity.value = p.shellIntensity;
  });

  return (
    <points geometry={geometry} frustumCulled={false}>
      <shaderMaterial
        vertexShader={VERTEX}
        fragmentShader={FRAGMENT}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
        toneMapped={false}
      />
    </points>
  );
}
