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
let gui = null;
let stats = { fps: 0, frameTime: 0 };


// Advanced color palette system
const COLOR_PALETTES = {
  rainbow: [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0.5, 0, 1]],
  fire: [[1, 1, 0], [1, 0.5, 0], [1, 0, 0], [0.8, 0, 0], [0.5, 0, 0]],
  ice: [[0, 1, 1], [0, 0.8, 1], [0, 0.5, 1], [0, 0.2, 1], [0, 0, 1]],
  toxic: [[0, 1, 0], [0.5, 1, 0], [1, 1, 0], [1, 0.5, 0], [1, 0, 0.5]],
  galaxy: [[0.5, 0, 1], [0, 0.5, 1], [0, 1, 1], [1, 0, 1], [1, 0.5, 0.8]],
  sunset: [[1, 0.3, 0], [1, 0.6, 0], [1, 0.8, 0.2], [0.8, 0.4, 0.8], [0.5, 0.2, 0.8]],
  ocean: [[0, 0.2, 0.8], [0, 0.6, 1], [0.2, 0.8, 1], [0.6, 1, 1], [0.8, 1, 0.8]],
  neon: [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0.5], [0.5, 1, 0]],
  pastel: [[1, 0.7, 0.8], [0.8, 0.9, 1], [0.9, 1, 0.7], [1, 0.9, 0.6], [0.9, 0.8, 1]]
};

// Color generation modes
const COLOR_MODES = {
  palette: 'Palette Based',
  temperature: 'Temperature',
  velocity: 'Velocity Based',
  duration: 'Click Duration',
  pressure: 'Pressure Based',
  rainbow: 'Rainbow Cycle',
  complementary: 'Complementary'
};

// Advanced color settings
let colorSettings = {
  mode: 'duration',
  temperatureRange: [2000, 8000], // Kelvin
  saturationBoost: 1.0,
  hueShift: 0.0,
  mixingStrength: 0.5,
  fadeRate: 0.02,
  chromaticAberration: 0.0
};

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
  if (!gui) setupGUI();
  if (prefersReducedMotion.matches) running = false;
  
  seedDye(); // Re-enabled with smaller, less intrusive seeds
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

function setupGUI() {
  if (!window.dat || !sim) return;
  
  gui = new dat.GUI({ width: 300 });
  gui.domElement.style.position = 'fixed';
  gui.domElement.style.top = '10px';
  gui.domElement.style.left = '20px'; // Move main GUI to the left side
  gui.domElement.style.zIndex = '1000';
  gui.domElement.style.pointerEvents = 'auto'; // Ensure GUI handles its own events

  // Performance folder
  const perfFolder = gui.addFolder('Performance');
  const perfStats = { 
    fps: 0, 
    frameTime: 0,
    showStats: true,
    autoQuality: true 
  };
  perfFolder.add(sim.settings, 'qualityScale', 0.25, 1.0).step(0.05).name('Quality Scale');
  perfFolder.add(perfStats, 'autoQuality').name('Auto Quality').onChange(v => {
    if (!v) sim.settings.qualityScale = 0.75;
  });
  const fpsDisplay = perfFolder.add(perfStats, 'fps').name('FPS').listen();
  const frameDisplay = perfFolder.add(perfStats, 'frameTime').name('Frame (ms)').listen();
  perfFolder.add(perfStats, 'showStats').name('Show Stats');
  perfFolder.open();

  // Simulation folder
  const simFolder = gui.addFolder('Fluid Simulation');
  simFolder.add(sim.settings, 'dyeDissipation', 0.9, 1.0).step(0.001).name('Dye Persistence');
  simFolder.add(sim.settings, 'velDissipation', 0.95, 1.0).step(0.001).name('Velocity Persistence');
  simFolder.add(sim.settings, 'pressureIters', 1, 50).step(1).name('Pressure Iterations');
  simFolder.add(sim.settings, 'vorticity', 0.0, 5.0).step(0.1).name('Vorticity Strength');
  simFolder.add(sim.settings, 'splatRadius', 0.005, 0.05).step(0.001).name('Splat Radius');

  // Visual Effects folder
  const fxFolder = gui.addFolder('Visual Effects');
  fxFolder.add(sim.settings, 'bloomEnabled').name('Enable Bloom').onChange(v => {
    bloomFolder.__ul.style.display = v ? 'block' : 'none';
  });
  
  const bloomFolder = fxFolder.addFolder('Bloom Settings');
  bloomFolder.add(sim.settings, 'bloomThreshold', 0.0, 1.5).step(0.01).name('Threshold');
  bloomFolder.add(sim.settings, 'bloomKnee', 0.0, 1.0).step(0.01).name('Knee');
  bloomFolder.add(sim.settings, 'bloomIntensity', 0.0, 2.0).step(0.01).name('Intensity');
  bloomFolder.add(sim.settings, 'bloomSaturation', 0.0, 2.0).step(0.01).name('Saturation');
  
  const blendModes = { Additive: 0, Screen: 1, Overlay: 2, 'Soft Light': 3 };
  bloomFolder.add(sim.settings, 'bloomBlendMode', blendModes).name('Blend Mode');
  bloomFolder.add(sim.settings, 'detailPreservation', 0.0, 1.0).step(0.01).name('Detail Preservation');
  bloomFolder.add(sim.settings, 'baseTexture', 0.0, 0.5).step(0.01).name('Base Texture Mix');

  // Advanced Rendering folder  
  const renderFolder = gui.addFolder('Advanced Rendering');
  renderFolder.add(sim.settings, 'exposure', -3.0, 3.0).step(0.01).name('Exposure');
  renderFolder.add(sim.settings, 'whitePoint', 0.5, 5.0).step(0.01).name('White Point');
  
  const tonemapModes = { Reinhard: 0, ACES: 1, Filmic: 2 };
  renderFolder.add(sim.settings, 'tonemapMode', tonemapModes).name('Tone Mapping');

  // Color folder
  const colorFolder = gui.addFolder('Color & Style');
  colorFolder.add(sim.settings, 'brightness', -0.5, 0.5).step(0.01).name('Brightness');
  colorFolder.add(sim.settings, 'contrast', 0.5, 2.0).step(0.01).name('Contrast');
  
  const colorBalance = { r: 1.0, g: 1.0, b: 1.0 };
  colorFolder.add(colorBalance, 'r', 0.0, 2.0).step(0.01).name('Red Balance').onChange(v => {
    sim.settings.colorBalance[0] = v;
  });
  colorFolder.add(colorBalance, 'g', 0.0, 2.0).step(0.01).name('Green Balance').onChange(v => {
    sim.settings.colorBalance[1] = v;
  });
  colorFolder.add(colorBalance, 'b', 0.0, 2.0).step(0.01).name('Blue Balance').onChange(v => {
    sim.settings.colorBalance[2] = v;
  });

  const paletteControl = { 
    palette: 'rainbow',
    speed: 1.0
  };
  
  // Color generation mode
  colorFolder.add(colorSettings, 'mode', Object.keys(COLOR_MODES)).name('Color Mode');
  
  // Advanced color controls
  const advancedFolder = colorFolder.addFolder('Advanced Colors');
  advancedFolder.add(colorSettings, 'hueShift', -1.0, 1.0).step(0.01).name('Hue Shift');
  advancedFolder.add(colorSettings, 'saturationBoost', 0.0, 2.0).step(0.01).name('Saturation Boost');
  
  const tempFolder = advancedFolder.addFolder('Temperature');
  tempFolder.add(colorSettings.temperatureRange, '0', 1000, 10000).step(100).name('Min Kelvin');
  tempFolder.add(colorSettings.temperatureRange, '1', 1000, 10000).step(100).name('Max Kelvin');
  
  // Palette controls
  colorFolder.add(paletteControl, 'palette', Object.keys(COLOR_PALETTES)).name('Color Palette').onChange(v => {
    sim.settings.colorPalette = v;
  });
  colorFolder.add(paletteControl, 'speed', 0.1, 3.0).step(0.1).name('Palette Speed').onChange(v => {
    sim.settings.paletteSpeed = v;
  });

  // Controls folder
  const ctrlFolder = gui.addFolder('Controls');
  const controls = { 
    pause: () => { running = !running; },
    reset: () => { sim.reset(); seedDye(); },
    clearDye: () => { sim.reset(); },
    addSeeds: () => { seedDye(); }
  };
  ctrlFolder.add(controls, 'pause').name('Pause/Resume');
  ctrlFolder.add(controls, 'reset').name('Reset Simulation');
  ctrlFolder.add(controls, 'clearDye').name('Clear Dye');
  ctrlFolder.add(controls, 'addSeeds').name('Add Seed Dye');

  // Update performance stats
  const updateStats = () => {
    perfStats.fps = Math.round(1000 / emaFrame);
    perfStats.frameTime = Math.round(emaFrame * 10) / 10;
    
    if (perfStats.showStats && perfStats.autoQuality) {
      adjustQuality(emaFrame);
    }
  };
  
  // Update stats every 100ms
  setInterval(updateStats, 100);
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

// Remove any existing event listeners to prevent duplicates
function removeEvents() {
  if (eventsAttached) {
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
  }
}

function attachEvents() {
  // Remove any existing listeners first
  removeEvents();
  
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
  canvas.addEventListener("webglcontextrestored", () => { 
    try { 
      sim = new FluidSim(canvas); 
      seedDye(); 
      running = true; 
    } catch (e) { 
      console.error('Context restore failed:', e); 
    } 
  });

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
  // Only handle pointer events on the canvas
  if (e.target !== canvas) return;
  
  const [x, y] = canvasSpace(e);
  canvas.setPointerCapture?.(e.pointerId);
  const startTime = performance.now();
  activePointers.set(e.pointerId, {
    lastX: x, lastY: y, lastT: startTime,
    startTime: startTime,
    pressure: e.pressure || 0.5,
    totalDistance: 0,
    color: pickColor(e.pointerId, 0, 0.5), // duration=0, pressure=0.5
  });
  splatAt(x, y, 0, 0, 0, activePointers.get(e.pointerId).color);
}

function onPointerMove(e) {
  if (!activePointers.has(e.pointerId)) return;
  
  // Prevent GUI interference
  if (e.target !== canvas) return;
  
  const p = activePointers.get(e.pointerId);
  const [x, y] = canvasSpace(e);
  const t = performance.now();
  const dt = Math.max(1e-3, (t - p.lastT) / 1000);
  const vx = (x - p.lastX) / dt;
  const vy = (y - p.lastY) / dt;
  
  // Update tracking data
  const distance = Math.sqrt((x - p.lastX) ** 2 + (y - p.lastY) ** 2);
  p.totalDistance += distance;
  p.pressure = e.pressure || p.pressure;
  const duration = (t - p.startTime) / 1000; // seconds
  const velocity = Math.sqrt(vx * vx + vy * vy);
  
  // Generate new color based on current parameters
  p.color = pickColor(e.pointerId, duration, p.pressure, velocity, p.totalDistance);
  
  // Color changes dynamically during movement based on duration, pressure, velocity
  
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

function pickColor(seed, duration = 0, pressure = 0.5, velocity = 0, totalDistance = 0) {
  const time = performance.now() * 0.001;
  let color = [1, 1, 1]; // Default white
  
  switch (colorSettings.mode) {
    case 'duration':
      // Color changes based on how long the click is held
      const durationHue = (duration * 0.3 + seed * 0.1) % 1;
      const durationSat = Math.min(0.8, 0.3 + duration * 0.1);
      const durationVal = Math.min(1.0, 0.7 + duration * 0.1);
      color = hsv2rgb(durationHue, durationSat, durationVal);
      break;
      
    case 'pressure':
      // Color intensity based on pressure/force
      const pressureHue = (seed * 0.61803398875 + pressure * 0.2) % 1;
      const pressureSat = 0.4 + pressure * 0.6;
      const pressureVal = 0.5 + pressure * 0.5;
      color = hsv2rgb(pressureHue, pressureSat, pressureVal);
      break;
      
    case 'velocity':
      // Color based on movement speed
      const velocityNorm = Math.min(1.0, velocity * 0.1);
      const velocityHue = (seed * 0.5 + velocityNorm * 0.3) % 1;
      const velocitySat = 0.6 + velocityNorm * 0.4;
      color = hsv2rgb(velocityHue, velocitySat, 0.9);
      break;
      
    case 'temperature':
      // Color temperature simulation
      const tempRange = colorSettings.temperatureRange[1] - colorSettings.temperatureRange[0];
      const temp = colorSettings.temperatureRange[0] + (duration * tempRange * 0.1) % tempRange;
      color = temperatureToRGB(temp);
      break;
      
    case 'rainbow':
      // Cycling rainbow
      const rainbowHue = (time * sim.settings.paletteSpeed + seed * 0.1) % 1;
      color = hsv2rgb(rainbowHue, 0.8, 0.95);
      break;
      
    case 'complementary':
      // Complementary color pairs
      const baseHue = (seed * 0.61803398875) % 1;
      const compHue = duration > 0.5 ? (baseHue + 0.5) % 1 : baseHue;
      color = hsv2rgb(compHue, 0.7 + Math.sin(duration * 3) * 0.2, 0.9);
      break;
      
    case 'palette':
    default:
      // Palette-based with enhancements
      if (sim && sim.settings.colorPalette && COLOR_PALETTES[sim.settings.colorPalette]) {
        const palette = COLOR_PALETTES[sim.settings.colorPalette];
        const paletteTime = time * sim.settings.paletteSpeed + duration * 0.5;
        const paletteIndex = ((seed * 17.32) + paletteTime * 0.5) % 1;
        const scaledIndex = paletteIndex * (palette.length - 1);
        const index1 = Math.floor(scaledIndex);
        const index2 = (index1 + 1) % palette.length;
        const t = scaledIndex - index1;
        
        const color1 = palette[index1];
        const color2 = palette[index2];
        
        color = [
          color1[0] * (1 - t) + color2[0] * t,
          color1[1] * (1 - t) + color2[1] * t,
          color1[2] * (1 - t) + color2[2] * t
        ];
      } else {
        // Fallback
        const h = (seed * 0.61803398875) % 1;
        color = hsv2rgb(h, 0.6, 0.95);
      }
      break;
  }
  
  // Apply global color modifications
  if (colorSettings.hueShift !== 0) {
    const hsv = rgb2hsv(color[0], color[1], color[2]);
    hsv[0] = (hsv[0] + colorSettings.hueShift) % 1;
    color = hsv2rgb(hsv[0], hsv[1], hsv[2]);
  }
  
  if (colorSettings.saturationBoost !== 1.0) {
    const hsv = rgb2hsv(color[0], color[1], color[2]);
    hsv[1] = Math.min(1.0, hsv[1] * colorSettings.saturationBoost);
    color = hsv2rgb(hsv[0], hsv[1], hsv[2]);
  }
  
  return color;
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

function rgb2hsv(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const diff = max - min;
  const v = max;
  const s = max === 0 ? 0 : diff / max;
  let h = 0;
  
  if (diff !== 0) {
    if (max === r) h = (g - b) / diff + (g < b ? 6 : 0);
    else if (max === g) h = (b - r) / diff + 2;
    else h = (r - g) / diff + 4;
    h /= 6;
  }
  
  return [h, s, v];
}

function temperatureToRGB(kelvin) {
  // Simplified blackbody radiation color temperature conversion
  const temp = kelvin / 100;
  let r, g, b;
  
  if (temp <= 66) {
    r = 255;
    g = temp > 19 ? 99.4708025861 * Math.log(temp - 10) - 161.1195681661 : 0;
    b = temp >= 20 ? 138.5177312231 * Math.log(temp - 10) - 305.0447927307 : 0;
  } else {
    r = 329.698727446 * Math.pow(temp - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(temp - 60, -0.0755148492);
    b = 255;
  }
  
  return [
    Math.max(0, Math.min(1, r / 255)),
    Math.max(0, Math.min(1, g / 255)),
    Math.max(0, Math.min(1, b / 255))
  ];
}

function seedDye() {
  if (!sim) return;
  const seeds = 3; // Reduced from 6 to 3
  for (let i = 0; i < seeds; i++) {
    const x = Math.random() * 0.6 + 0.2; // Smaller area
    const y = Math.random() * 0.6 + 0.2; // Smaller area
    const c = pickColor(i + 1);
    sim.splat([x, y], c, sim.settings.splatRadius * 1.5, "dye"); // Smaller splat
  }
}

// Kick things off
init();
