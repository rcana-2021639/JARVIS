/**
 * Capa 6 — postproceso. Sin esto no hay look: el bloom es el 60 % del resultado.
 *
 *   bloom                umbral bajo, intensidad alta, solo sobre lo emisivo
 *   aberracion cromatica muy sutil, la del borde de un lente
 *   vineta               el centro manda
 *   grano                al 3 %, para que la luz tenga textura de pelicula
 */

import { Bloom, ChromaticAberration, EffectComposer, Noise, Vignette } from "@react-three/postprocessing";
import { BlendFunction } from "postprocessing";
import { useMemo } from "react";
import { Vector2 } from "three";

export interface EffectsProps {
  /** En el nivel de detalle mas bajo, el bloom usa menos pasadas. */
  readonly lite: boolean;
}

export function Effects({ lite }: EffectsProps): JSX.Element {
  const aberration = useMemo(() => new Vector2(0.0007, 0.0005), []);
  return (
    <EffectComposer multisampling={0} enableNormalPass={false}>
      <Bloom
        intensity={lite ? 1.5 : 1.9}
        luminanceThreshold={0.05}
        luminanceSmoothing={0.4}
        mipmapBlur
        radius={lite ? 0.6 : 0.78}
        levels={lite ? 5 : 8}
      />
      <ChromaticAberration
        offset={aberration}
        radialModulation={false}
        modulationOffset={0}
        blendFunction={BlendFunction.NORMAL}
      />
      <Vignette offset={0.26} darkness={0.82} />
      <Noise opacity={0.03} premultiply blendFunction={BlendFunction.SCREEN} />
    </EffectComposer>
  );
}
