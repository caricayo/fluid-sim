"use strict";

import { FluidSim } from "./sim.js";

const canvas = document.getElementById("sim");
const ui = document.getElementById("ui");

const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)") || { matches: false, addEventListener() {} };

let sim = null;
let running = true;
let lastT = performance.now();
let emaFrame = 16.6; // ms
let eventsAttached = false;

function init() {
  try {
    sim = new FluidSim(canvas, {
      qualityScale: 0.75,
      vorticity: 1.2,
      pressureIters: 22,
      dyeDissipation: 0.992,
    });
  } catch (e) {
    console.error(e);
    showError(String(e?.message || e));
    return;
  }

  if (!eventsAttached) attachEvents();
  if (prefersReducedMotion.matches) running = false;
  seedDye();
  requestAnimationFrame(tick);
}

function showError(message) {
  const msg = document.createElement("div");
  msg.textContent = `WebGL Error: ${message}`;
  msg.style.position = "fixed";
  msg.style.left = "50%";
  msg.style.top = "50%";
  msg.style.transform = "translate(-50%, -50%)";
  msg.style.background = "#1a1d22";
  msg.style.color = "#fff";
  msg.style.padding = "12px 16px";
  msg.style.borderRadius = "10px";
  msg.style.border = "1px solid rgba(255,255,255,0.12)";
  msg.style.font = "600 13px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu";
  msg.style.pointerEvents = "none";
  ui?.appendChild(msg);
}

function tick(now) {
  const dt = Math.min(0.033, Math.max(0.0, (now - lastT) / 1000));
  lastT = now;
  if (running && sim) {
    sim._resize();
    sim.step(dt);
    sim.renderToScreen();
  }
  // dynamic quality scaling (gentle)
  emaFrame = 0.9 * emaFrame + 0.1 * (dt * 1000);
  if (sim) adjustQuality(emaFrame);
  requestAnimationFrame(tick);
}

function adjustQuality(ms) {
  // Target ~16.6ms; adjust sim scale slightly
  const s = sim.settings;
  if (ms > 19 && s.qualityScale > 0.45) {
    s.qualityScale = Math.max(0.45, s.qualityScale - 0.02);
    sim._resize();
  } else if (ms < 14 && s.qualityScale < 1.0) {
    s.qualityScale = Math.min(1.0, s.qualityScale + 0.02);
    sim._resize();
  }
}

// Interaction: Pointer events with multi-touch splats
const activePointers = new Map();
function attachEvents() {
  window.addEventListener("resize", () => sim?._resize());
  document.addEventListener("visibilitychange", () => {
    running = document.visibilityState === "visible" && !prefersReducedMotion.matches;
  });
  if (prefersReducedMotion.addEventListener) {
    prefersReducedMotion.addEventListener("change", (e) => {
      running = !e.matches;
    });
  }

  canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); running = false; });
  canvas.addEventListener("webglcontextrestored", () => { try { sim = new FluidSim(canvas); seedDye(); running = true; } catch { /* noop */ } });

  canvas.addEventListener("pointerdown", onPointerDown, { passive: true });
  canvas.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerup", onPointerUp, { passive: true });

  window.addEventListener("keydown", (e) => {
    if (!sim) return;
    const k = e.key.toLowerCase();
    if (k === "p") { running = !running; }
    if (k === "q") { cycleQuality(); }
    if (k === "v") { sim.settings.vorticity = sim.settings.vorticity > 0 ? 0.0 : 1.2; }
    if (k === "r") { sim.reset(); }
  });
  eventsAttached = true;
}

function cycleQuality() {
  const steps = [0.5, 0.75, 1.0];
  const s = sim.settings;
  const i = steps.findIndex(v => Math.abs(v - s.qualityScale) < 1e-3);
  const next = steps[(i + 1) % steps.length];
  s.qualityScale = next;
  sim._resize();
}

function canvasSpace(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  return [x, 1 - y]; // flip y for GL
}

function onPointerDown(e) {
  canvas.setPointerCapture?.(e.pointerId);
  const [x, y] = canvasSpace(e);
  activePointers.set(e.pointerId, {
    lastX: x, lastY: y, lastT: performance.now(),
    color: pickColor(e.pointerId),
  });
  splatAt(x, y, 0, 0, 0, activePointers.get(e.pointerId).color);
}

function onPointerMove(e) {
  if (!activePointers.has(e.pointerId)) return;
  const p = activePointers.get(e.pointerId);
  const [x, y] = canvasSpace(e);
  const t = performance.now();
  const dt = Math.max(1e-3, (t - p.lastT) / 1000);
  const vx = (x - p.lastX) / dt;
  const vy = (y - p.lastY) / dt;
  p.lastX = x; p.lastY = y; p.lastT = t;
  splatAt(x, y, vx, vy, 0, p.color);
}

function onPointerUp(e) {
  activePointers.delete(e.pointerId);
}

function splatAt(x, y, vx, vy, _vz, color) {
  if (!sim) return;
  const r = sim.settings.splatRadius * (window.devicePixelRatio || 1);
  // Velocity splat scaled by pointer speed
  sim.splat([x, y], [vx * 0.25, vy * 0.25, 0], r, "velocity");
  // Dye splat
  sim.splat([x, y], color, r, "dye");
}

function pickColor(seed) {
  // Deterministic pretty color
  const h = (seed * 0.61803398875) % 1;
  const s = 0.6, v = 0.95;
  const rgb = hsv2rgb(h, s, v);
  return rgb;
}

function hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
    default: return [v, t, p];
  }
}

function seedDye() {
  if (!sim) return;
  const seeds = 6;
  for (let i = 0; i < seeds; i++) {
    const x = Math.random() * 0.8 + 0.1;
    const y = Math.random() * 0.8 + 0.1;
    const c = pickColor(i + 1);
    sim.splat([x, y], c, sim.settings.splatRadius * 2.5, "dye");
  }
}

// Kick things off
init();
