/**
 * Tokens de diseno de JARVIS: la UNICA fuente de verdad.
 *
 * Los mismos valores sirven como constantes de TypeScript (shaders, Three.js)
 * y como variables CSS (`applyTokens` las escribe en :root al arrancar). Nadie
 * escribe un color ni una duracion a mano fuera de este archivo.
 */

export const color = {
  void: "#000308",
  abyss: "#050B12",
  deep: "#0A1520",
  cyanCore: "#00E5FF",
  cyanGlow: "#7DF9FF",
  cyanDim: "#0A5A6B",
  amberCore: "#FFA726",
  amberHot: "#FFD54F",
  crimson: "#FF3B30",
  mint: "#00FFA3",
  textPrimary: "#DFF7FF",
  textMuted: "#4A7A8C",
  /** El violeta de PENSANDO: el documento lo nombra ("violeta-cian") sin dar valor. */
  violet: "#7C4DFF",
} as const;

export const font = {
  display: "'Orbitron', 'Michroma', sans-serif",
  mono: "'JetBrains Mono', ui-monospace, monospace",
  ui: "'Rajdhani', 'Chakra Petch', sans-serif",
} as const;

/** Escala tipografica en px. */
export const fontSize = [11, 13, 16, 20, 28, 42] as const;
export const tracking = "0.08em";

export const geometry = {
  border: 1,
  grid: 8,
  /** Tamano del corte diagonal de las esquinas (chamfer). */
  chamfer: 16,
  panelBorderAlpha: 0.3,
} as const;

export const motion = {
  /** cubic-bezier(0.16, 1, 0.3, 1) */
  curve: [0.16, 1, 0.3, 1] as const,
  microMs: 120,
  transitionMs: 280,
  cinematicMs: 800,
  breathMs: 4000,
} as const;

export type ColorToken = keyof typeof color;

function kebab(name: string): string {
  return name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

/** Las variables CSS que se escriben en :root, derivadas de las constantes. */
export function cssVariables(): Record<string, string> {
  const vars: Record<string, string> = {};
  for (const [name, value] of Object.entries(color)) {
    vars[`--${kebab(name)}`] = value;
  }
  vars["--font-display"] = font.display;
  vars["--font-mono"] = font.mono;
  vars["--font-ui"] = font.ui;
  fontSize.forEach((size, index) => {
    vars[`--fs-${index}`] = `${size}px`;
  });
  vars["--tracking"] = tracking;
  vars["--border"] = `${geometry.border}px`;
  vars["--grid"] = `${geometry.grid}px`;
  vars["--chamfer"] = `${geometry.chamfer}px`;
  vars["--ease"] = `cubic-bezier(${motion.curve.join(", ")})`;
  vars["--t-micro"] = `${motion.microMs}ms`;
  vars["--t-transition"] = `${motion.transitionMs}ms`;
  vars["--t-cinematic"] = `${motion.cinematicMs}ms`;
  vars["--t-breath"] = `${motion.breathMs}ms`;
  return vars;
}

export function applyTokens(root: HTMLElement): void {
  for (const [name, value] of Object.entries(cssVariables())) {
    root.style.setProperty(name, value);
  }
}

/**
 * La curva estandar evaluada en t (0..1): el mismo cubic-bezier del CSS, para
 * interpolar estados del nucleo en los shaders con la misma sensacion.
 */
export function ease(t: number): number {
  const [x1, y1, x2, y2] = motion.curve;
  const clamped = Math.min(1, Math.max(0, t));
  // Newton-Raphson sobre x(s) = t, luego y(s).
  let s = clamped;
  for (let i = 0; i < 6; i += 1) {
    const x = bezier(s, x1, x2) - clamped;
    const dx = bezierSlope(s, x1, x2);
    if (Math.abs(dx) < 1e-6) break;
    s -= x / dx;
  }
  return bezier(Math.min(1, Math.max(0, s)), y1, y2);
}

function bezier(s: number, p1: number, p2: number): number {
  const inv = 1 - s;
  return 3 * inv * inv * s * p1 + 3 * inv * s * s * p2 + s * s * s;
}

function bezierSlope(s: number, p1: number, p2: number): number {
  const inv = 1 - s;
  return 3 * inv * inv * p1 + 6 * inv * s * (p2 - p1) + 3 * s * s * (1 - p2);
}
