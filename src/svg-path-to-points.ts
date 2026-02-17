/**
 * SVG path `d` string → Excalidraw freedraw points converter.
 *
 * Parses all SVG path commands (M/m, L/l, H/h, V/v, C/c, S/s, Q/q, T/t, A/a, Z/z),
 * samples curves adaptively using de Casteljau subdivision, and normalizes
 * the resulting points so the minimum is at (0, 0).
 */

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

const CMD_RE = /([MmLlHhVvCcSsQqTtAaZz])/;

interface Token {
  cmd: string;
  args: number[];
}

function tokenize(d: string): Token[] {
  // Split on command letters, keeping the delimiter
  const parts = d.split(CMD_RE).filter(Boolean);
  const tokens: Token[] = [];
  let i = 0;
  while (i < parts.length) {
    const cmd = parts[i];
    if (!CMD_RE.test(cmd)) { i++; continue; }
    i++;
    const argStr = i < parts.length && !CMD_RE.test(parts[i]) ? parts[i++] : "";
    // Parse numbers: handle negatives, decimals, and comma/space separators
    const args = (argStr.match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g) || []).map(Number);
    tokens.push({ cmd, args });
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Cubic Bezier adaptive sampling (de Casteljau)
// ---------------------------------------------------------------------------

type Pt = [number, number];

function flatness(p0: Pt, p1: Pt, p2: Pt, p3: Pt): number {
  // Maximum deviation of control points from the line p0→p3
  const ux = 3 * p1[0] - 2 * p0[0] - p3[0];
  const uy = 3 * p1[1] - 2 * p0[1] - p3[1];
  const vx = 3 * p2[0] - 2 * p3[0] - p0[0];
  const vy = 3 * p2[1] - 2 * p3[1] - p0[1];
  return Math.max(ux * ux, vx * vx) + Math.max(uy * uy, vy * vy);
}

const FLATNESS_THRESHOLD = 0.5 * 0.5; // 0.5px squared
const MAX_DEPTH = 6;

function sampleCubic(p0: Pt, p1: Pt, p2: Pt, p3: Pt, points: Pt[], depth: number): void {
  if (depth >= MAX_DEPTH || flatness(p0, p1, p2, p3) < FLATNESS_THRESHOLD) {
    points.push(p3);
    return;
  }
  // de Casteljau split at t=0.5
  const m01: Pt = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
  const m12: Pt = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
  const m23: Pt = [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2];
  const m012: Pt = [(m01[0] + m12[0]) / 2, (m01[1] + m12[1]) / 2];
  const m123: Pt = [(m12[0] + m23[0]) / 2, (m12[1] + m23[1]) / 2];
  const mid: Pt = [(m012[0] + m123[0]) / 2, (m012[1] + m123[1]) / 2];
  sampleCubic(p0, m01, m012, mid, points, depth + 1);
  sampleCubic(mid, m123, m23, p3, points, depth + 1);
}

// ---------------------------------------------------------------------------
// Quadratic Bezier → sample via cubic promotion
// ---------------------------------------------------------------------------

function quadToCubic(p0: Pt, cp: Pt, p2: Pt): [Pt, Pt, Pt, Pt] {
  // Promote quadratic to cubic: cp1 = p0 + 2/3*(cp-p0), cp2 = p2 + 2/3*(cp-p2)
  return [
    p0,
    [p0[0] + (2 / 3) * (cp[0] - p0[0]), p0[1] + (2 / 3) * (cp[1] - p0[1])],
    [p2[0] + (2 / 3) * (cp[0] - p2[0]), p2[1] + (2 / 3) * (cp[1] - p2[1])],
    p2,
  ];
}

// ---------------------------------------------------------------------------
// Arc (A/a) → cubic Bezier approximation
// ---------------------------------------------------------------------------

function arcToCubics(
  x1: number, y1: number,
  rx: number, ry: number,
  xAxisRotation: number,
  largeArcFlag: number,
  sweepFlag: number,
  x2: number, y2: number,
): [Pt, Pt, Pt, Pt][] {
  // Implementation based on the SVG spec arc parameterization
  if (rx === 0 || ry === 0) return []; // degenerate → line (handled by caller)

  rx = Math.abs(rx);
  ry = Math.abs(ry);
  const phi = (xAxisRotation * Math.PI) / 180;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);

  // Step 1: compute (x1', y1')
  const dx2 = (x1 - x2) / 2;
  const dy2 = (y1 - y2) / 2;
  const x1p = cosPhi * dx2 + sinPhi * dy2;
  const y1p = -sinPhi * dx2 + cosPhi * dy2;

  // Step 2: compute (cx', cy')
  let rxSq = rx * rx;
  let rySq = ry * ry;
  const x1pSq = x1p * x1p;
  const y1pSq = y1p * y1p;

  // Adjust radii if too small
  const lambda = x1pSq / rxSq + y1pSq / rySq;
  if (lambda > 1) {
    const sqrtLambda = Math.sqrt(lambda);
    rx *= sqrtLambda;
    ry *= sqrtLambda;
    rxSq = rx * rx;
    rySq = ry * ry;
  }

  let sq = Math.max(0, (rxSq * rySq - rxSq * y1pSq - rySq * x1pSq) / (rxSq * y1pSq + rySq * x1pSq));
  sq = Math.sqrt(sq);
  if (largeArcFlag === sweepFlag) sq = -sq;
  const cxp = sq * (rx * y1p) / ry;
  const cyp = sq * -(ry * x1p) / rx;

  // Step 3: compute (cx, cy) from (cx', cy')
  const cx = cosPhi * cxp - sinPhi * cyp + (x1 + x2) / 2;
  const cy = sinPhi * cxp + cosPhi * cyp + (y1 + y2) / 2;

  // Step 4: compute theta1 and dTheta
  const vAngle = (ux: number, uy: number, vx: number, vy: number): number => {
    const dot = ux * vx + uy * vy;
    const len = Math.sqrt(ux * ux + uy * uy) * Math.sqrt(vx * vx + vy * vy);
    let ang = Math.acos(Math.max(-1, Math.min(1, dot / len)));
    if (ux * vy - uy * vx < 0) ang = -ang;
    return ang;
  };

  const theta1 = vAngle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
  let dTheta = vAngle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry);

  if (sweepFlag === 0 && dTheta > 0) dTheta -= 2 * Math.PI;
  if (sweepFlag === 1 && dTheta < 0) dTheta += 2 * Math.PI;

  // Split into segments of at most PI/2
  const segments = Math.ceil(Math.abs(dTheta) / (Math.PI / 2));
  const segAngle = dTheta / segments;
  const cubics: [Pt, Pt, Pt, Pt][] = [];

  let curTheta = theta1;
  let curX = x1;
  let curY = y1;

  for (let i = 0; i < segments; i++) {
    const nextTheta = curTheta + segAngle;
    // Approximate arc segment with cubic Bezier
    const alpha = Math.sin(segAngle) * (Math.sqrt(4 + 3 * Math.tan(segAngle / 2) ** 2) - 1) / 3;

    const cosT1 = Math.cos(curTheta);
    const sinT1 = Math.sin(curTheta);
    const cosT2 = Math.cos(nextTheta);
    const sinT2 = Math.sin(nextTheta);

    const ep1x = cosPhi * rx * cosT1 - sinPhi * ry * sinT1 + cx;
    const ep1y = sinPhi * rx * cosT1 + cosPhi * ry * sinT1 + cy;
    const ep2x = cosPhi * rx * cosT2 - sinPhi * ry * sinT2 + cx;
    const ep2y = sinPhi * rx * cosT2 + cosPhi * ry * sinT2 + cy;

    const d1x = -cosPhi * rx * sinT1 - sinPhi * ry * cosT1;
    const d1y = -sinPhi * rx * sinT1 + cosPhi * ry * cosT1;
    const d2x = -cosPhi * rx * sinT2 - sinPhi * ry * cosT2;
    const d2y = -sinPhi * rx * sinT2 + cosPhi * ry * cosT2;

    cubics.push([
      [curX, curY],
      [ep1x + alpha * d1x, ep1y + alpha * d1y],
      [ep2x - alpha * d2x, ep2y - alpha * d2y],
      [ep2x, ep2y],
    ]);

    curTheta = nextTheta;
    curX = ep2x;
    curY = ep2y;
  }

  return cubics;
}

// ---------------------------------------------------------------------------
// Main converter
// ---------------------------------------------------------------------------

export function svgPathToFreedraw(d: string): {
  points: [number, number][];
  width: number;
  height: number;
} {
  const tokens = tokenize(d);
  const points: Pt[] = [];

  let cx = 0, cy = 0; // current position
  let sx = 0, sy = 0; // subpath start (for Z)
  let prevCp: Pt | null = null; // previous control point (for S/T)
  let prevCmd = ""; // previous command letter

  for (const { cmd, args } of tokens) {
    const isRel = cmd === cmd.toLowerCase();

    switch (cmd) {
      // ---- Move ----
      case "M":
      case "m": {
        let i = 0;
        while (i < args.length) {
          const x = isRel ? cx + args[i] : args[i];
          const y = isRel ? cy + args[i + 1] : args[i + 1];
          cx = x; cy = y;
          if (i === 0) { sx = x; sy = y; }
          points.push([cx, cy]);
          i += 2;
          // Subsequent coordinate pairs after M are treated as implicit L
        }
        prevCp = null;
        break;
      }

      // ---- Line ----
      case "L":
      case "l": {
        for (let i = 0; i < args.length; i += 2) {
          cx = isRel ? cx + args[i] : args[i];
          cy = isRel ? cy + args[i + 1] : args[i + 1];
          points.push([cx, cy]);
        }
        prevCp = null;
        break;
      }

      // ---- Horizontal line ----
      case "H":
      case "h": {
        for (let i = 0; i < args.length; i++) {
          cx = isRel ? cx + args[i] : args[i];
          points.push([cx, cy]);
        }
        prevCp = null;
        break;
      }

      // ---- Vertical line ----
      case "V":
      case "v": {
        for (let i = 0; i < args.length; i++) {
          cy = isRel ? cy + args[i] : args[i];
          points.push([cx, cy]);
        }
        prevCp = null;
        break;
      }

      // ---- Cubic Bezier ----
      case "C":
      case "c": {
        for (let i = 0; i + 5 < args.length; i += 6) {
          const x1 = isRel ? cx + args[i] : args[i];
          const y1 = isRel ? cy + args[i + 1] : args[i + 1];
          const x2 = isRel ? cx + args[i + 2] : args[i + 2];
          const y2 = isRel ? cy + args[i + 3] : args[i + 3];
          const x = isRel ? cx + args[i + 4] : args[i + 4];
          const y = isRel ? cy + args[i + 5] : args[i + 5];
          sampleCubic([cx, cy], [x1, y1], [x2, y2], [x, y], points, 0);
          prevCp = [x2, y2];
          cx = x; cy = y;
        }
        break;
      }

      // ---- Smooth cubic Bezier ----
      case "S":
      case "s": {
        for (let i = 0; i + 3 < args.length; i += 4) {
          // Reflected control point
          const x1 = (prevCmd === "C" || prevCmd === "c" || prevCmd === "S" || prevCmd === "s") && prevCp
            ? 2 * cx - prevCp[0]
            : cx;
          const y1 = (prevCmd === "C" || prevCmd === "c" || prevCmd === "S" || prevCmd === "s") && prevCp
            ? 2 * cy - prevCp[1]
            : cy;
          const x2 = isRel ? cx + args[i] : args[i];
          const y2 = isRel ? cy + args[i + 1] : args[i + 1];
          const x = isRel ? cx + args[i + 2] : args[i + 2];
          const y = isRel ? cy + args[i + 3] : args[i + 3];
          sampleCubic([cx, cy], [x1, y1], [x2, y2], [x, y], points, 0);
          prevCp = [x2, y2];
          cx = x; cy = y;
        }
        break;
      }

      // ---- Quadratic Bezier ----
      case "Q":
      case "q": {
        for (let i = 0; i + 3 < args.length; i += 4) {
          const cpx = isRel ? cx + args[i] : args[i];
          const cpy = isRel ? cy + args[i + 1] : args[i + 1];
          const x = isRel ? cx + args[i + 2] : args[i + 2];
          const y = isRel ? cy + args[i + 3] : args[i + 3];
          const [p0, c1, c2, p3] = quadToCubic([cx, cy], [cpx, cpy], [x, y]);
          sampleCubic(p0, c1, c2, p3, points, 0);
          prevCp = [cpx, cpy];
          cx = x; cy = y;
        }
        break;
      }

      // ---- Smooth quadratic Bezier ----
      case "T":
      case "t": {
        for (let i = 0; i + 1 < args.length; i += 2) {
          const hasQuadPrev = (prevCmd === "Q" || prevCmd === "q" || prevCmd === "T" || prevCmd === "t") && prevCp;
          const cpx: number = hasQuadPrev ? 2 * cx - prevCp![0] : cx;
          const cpy: number = hasQuadPrev ? 2 * cy - prevCp![1] : cy;
          const x = isRel ? cx + args[i] : args[i];
          const y = isRel ? cy + args[i + 1] : args[i + 1];
          const [p0, c1, c2, p3] = quadToCubic([cx, cy], [cpx, cpy], [x, y]);
          sampleCubic(p0, c1, c2, p3, points, 0);
          prevCp = [cpx, cpy];
          cx = x; cy = y;
        }
        break;
      }

      // ---- Arc ----
      case "A":
      case "a": {
        for (let i = 0; i + 6 < args.length; i += 7) {
          const rx = args[i];
          const ry = args[i + 1];
          const xRot = args[i + 2];
          const largeArc = args[i + 3];
          const sweep = args[i + 4];
          const x = isRel ? cx + args[i + 5] : args[i + 5];
          const y = isRel ? cy + args[i + 6] : args[i + 6];

          if (rx === 0 || ry === 0) {
            // Degenerate arc → straight line
            cx = x; cy = y;
            points.push([cx, cy]);
          } else {
            const cubics = arcToCubics(cx, cy, rx, ry, xRot, largeArc, sweep, x, y);
            for (const [p0, c1, c2, p3] of cubics) {
              sampleCubic(p0, c1, c2, p3, points, 0);
            }
            cx = x; cy = y;
          }
          prevCp = null;
        }
        break;
      }

      // ---- Close path ----
      case "Z":
      case "z": {
        if (cx !== sx || cy !== sy) {
          points.push([sx, sy]);
        }
        cx = sx; cy = sy;
        prevCp = null;
        break;
      }
    }

    prevCmd = cmd;
  }

  // No points parsed
  if (points.length === 0) {
    return { points: [[0, 0]], width: 0, height: 0 };
  }

  // Compute bounding box and normalize to (0, 0) origin
  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  const normalized: [number, number][] = points.map(([x, y]) => [x - minX, y - minY]);
  const width = maxX - minX;
  const height = maxY - minY;

  return { points: normalized, width, height };
}
