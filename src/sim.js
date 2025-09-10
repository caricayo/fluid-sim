"use strict";

import { getGL, createProgram, createFullscreenVAO, checkFloatSupport, makeFBO, makeDoubleFBO, setViewport, bindFBO } from "./gl-helpers.js";

// GLSL Shaders
const VS_SCREEN = `#version 300 es
layout (location = 0) in vec2 a_pos;
void main(){
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Utility: read from sampler at pixel space via manual bilinear filter (works with NEAREST textures)
const GLSL_BILERP = `
vec4 bilerp(sampler2D tex, vec2 uv, vec2 resolution) {
  vec2 xy = uv * resolution - 0.5;
  vec2 ixy = floor(xy);
  vec2 f = xy - ixy;
  vec2 base = (ixy + 0.5) / resolution;
  vec2 texel = 1.0 / resolution;
  vec4 c00 = texture(tex, base);
  vec4 c10 = texture(tex, base + vec2(texel.x, 0.0));
  vec4 c01 = texture(tex, base + vec2(0.0, texel.y));
  vec4 c11 = texture(tex, base + texel);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}`;

// Advection (semi-Lagrangian) with dissipation
const FS_ADVECT = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_source; // field to advect (e.g., velocity or dye)
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
uniform vec2 u_resolution;
uniform float u_dt;
uniform float u_dissipation;
${GLSL_BILERP}
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec2 vel = texture(u_velocity, uv).xy;
  vec2 coord = uv - u_dt * vel * vec2(u_resolution.y / u_resolution.x, 1.0); // aspect compensate
  vec4 src = bilerp(u_source, coord, u_resolution);
  frag = src * u_dissipation;
}`;

// Divergence of velocity field
const FS_DIVERGENCE = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec2 L = texture(u_velocity, uv - vec2(u_texelSize.x, 0)).xy;
  vec2 R = texture(u_velocity, uv + vec2(u_texelSize.x, 0)).xy;
  vec2 B = texture(u_velocity, uv - vec2(0, u_texelSize.y)).xy;
  vec2 T = texture(u_velocity, uv + vec2(0, u_texelSize.y)).xy;
  float div = 0.5 * ((R.x - L.x) + (T.y - B.y));
  frag = vec4(div, 0.0, 0.0, 1.0);
}`;

// Jacobi pressure solve
const FS_PRESSURE = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = texture(u_pressure, uv - vec2(u_texelSize.x, 0)).x;
  float R = texture(u_pressure, uv + vec2(u_texelSize.x, 0)).x;
  float B = texture(u_pressure, uv - vec2(0, u_texelSize.y)).x;
  float T = texture(u_pressure, uv + vec2(0, u_texelSize.y)).x;
  float div = texture(u_divergence, uv).x;
  float p = (L + R + B + T - div) * 0.25;
  frag = vec4(p, 0.0, 0.0, 1.0);
}`;

// Subtract pressure gradient
const FS_GRADIENT = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = texture(u_pressure, uv - vec2(u_texelSize.x, 0)).x;
  float R = texture(u_pressure, uv + vec2(u_texelSize.x, 0)).x;
  float B = texture(u_pressure, uv - vec2(0, u_texelSize.y)).x;
  float T = texture(u_pressure, uv + vec2(0, u_texelSize.y)).x;
  vec2 v = texture(u_velocity, uv).xy;
  vec2 grad = 0.5 * vec2(R - L, T - B);
  frag = vec4(v - grad, 0.0, 1.0);
}`;

// Curl (vorticity) and confinement
const FS_CURL = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = texture(u_velocity, uv - vec2(u_texelSize.x, 0)).y;
  float R = texture(u_velocity, uv + vec2(u_texelSize.x, 0)).y;
  float B = texture(u_velocity, uv - vec2(0, u_texelSize.y)).x;
  float T = texture(u_velocity, uv + vec2(0, u_texelSize.y)).x;
  float curl = R - L - (T - B);
  frag = vec4(curl, 0.0, 0.0, 1.0);
}`;

const FS_VORTICITY = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform vec2 u_texelSize;
uniform float u_dt;
uniform float u_eps; // vorticity strength
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = abs(texture(u_curl, uv - vec2(u_texelSize.x, 0)).x);
  float R = abs(texture(u_curl, uv + vec2(u_texelSize.x, 0)).x);
  float B = abs(texture(u_curl, uv - vec2(0, u_texelSize.y)).x);
  float T = abs(texture(u_curl, uv + vec2(0, u_texelSize.y)).x);
  vec2 grad = vec2(R - L, T - B);
  float mag = length(grad) + 1e-5;
  vec2 N = grad / mag;
  float curl = texture(u_curl, uv).x;
  // Perpendicular to N in 2D for confinement force
  vec2 force = u_eps * vec2(N.y, -N.x) * curl;
  vec2 v = texture(u_velocity, uv).xy;
  v += force * u_dt;
  frag = vec4(v, 0.0, 1.0);
}`;

// Splat for dye or velocity
const FS_SPLAT = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_target;
uniform vec2 u_point; // in 0..1
uniform vec3 u_value; // rgb for dye; xy for velocity (z ignored)
uniform float u_radius; // in 0..1, roughly pixels proportion
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec2 d = uv - u_point;
  float r = u_radius;
  float a = exp(-dot(d, d) / (r*r + 1e-6));
  vec4 base = texture(u_target, uv);
  frag = base + vec4(u_value * a, 1.0);
}`;

// Bloom threshold - extract bright areas
const FS_BLOOM_PREFILTER = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_dye;
uniform vec2 u_texelSize;
uniform float u_threshold;
uniform float u_knee;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec3 c = texture(u_dye, uv).rgb;
  float brightness = max(c.r, max(c.g, c.b));
  float soft = brightness - u_threshold + u_knee;
  soft = clamp(soft, 0.0, 2.0 * u_knee);
  soft = soft * soft / (4.0 * u_knee + 1e-4);
  float contribution = max(soft, brightness - u_threshold);
  contribution /= max(brightness, 1e-4);
  frag = vec4(c * contribution, 1.0);
}`;

// Gaussian blur
const FS_BLUR = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_source;
uniform vec2 u_texelSize;
uniform vec2 u_direction;
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec3 result = texture(u_source, uv).rgb * weights[0];
  for(int i = 1; i < 5; ++i) {
    vec2 offset = float(i) * u_direction * u_texelSize;
    result += texture(u_source, uv + offset).rgb * weights[i];
    result += texture(u_source, uv - offset).rgb * weights[i];
  }
  frag = vec4(result, 1.0);
}`;

// Advanced bloom combine with multiple blending modes
const FS_BLOOM_FINAL = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_base;
uniform sampler2D u_bloom;
uniform vec2 u_texelSize;
uniform float u_intensity;
uniform float u_saturation;
uniform int u_blendMode; // 0=add, 1=screen, 2=overlay, 3=soft light
uniform float u_detailPreservation;
uniform float u_baseTexture;
uniform float u_brightness;
uniform float u_contrast;
uniform vec3 u_colorBalance;
uniform float u_exposure;
uniform float u_whitePoint;
uniform int u_tonemapMode;

vec3 screen(vec3 base, vec3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

vec3 overlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(0.5, base)
    );
}

vec3 softLight(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend + base * base * (1.0 - 2.0 * blend),
        sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend),
        step(0.5, blend)
    );
}

vec3 aces(vec3 color) {
    const float A = 2.51, B = 0.03, C = 2.43, D = 0.59, E = 0.14;
    return clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0);
}

vec3 reinhard(vec3 color) {
    return color / (1.0 + color);
}

vec3 filmic(vec3 color) {
    color = max(vec3(0.0), color - 0.004);
    return (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
}

void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  
  // Get raw base color and apply basic rendering
  vec3 base = texture(u_base, uv).rgb;
  
  // Apply color grading to base (same as in render shader)
  base *= u_brightness;
  base = (base - 0.5) * u_contrast + 0.5;
  base *= u_colorBalance;
  base *= pow(2.0, u_exposure);
  base /= u_whitePoint;
  
  // Apply tone mapping to base
  if (u_tonemapMode == 1) {
    base = aces(base);
  } else if (u_tonemapMode == 2) {
    base = reinhard(base);
  } else if (u_tonemapMode == 3) {
    base = filmic(base);
  }
  
  // Get bloom and enhance saturation
  vec3 bloom = texture(u_bloom, uv).rgb * u_intensity;
  float luminance = dot(bloom, vec3(0.299, 0.587, 0.114));
  bloom = mix(vec3(luminance), bloom, u_saturation);
  
  // Apply different blending modes
  vec3 result;
  if (u_blendMode == 1) {
      result = screen(base, bloom);
  } else if (u_blendMode == 2) {
      result = overlay(base, bloom);
  } else if (u_blendMode == 3) {
      result = softLight(base, bloom);
  } else {
      result = base + bloom; // Additive (default)
  }
  
  // Detail preservation - blend with original base
  float baseLuminance = dot(base, vec3(0.299, 0.587, 0.114));
  float preservation = u_detailPreservation * (1.0 - smoothstep(0.0, 1.0, baseLuminance));
  result = mix(result, base + bloom * 0.3, preservation);
  
  // Base texture preservation - always maintain some of the original texture
  // This ensures texture is visible even with maximum bloom intensity
  float baseAmount = u_baseTexture;
  vec3 finalResult = mix(result, base + result * 0.5, baseAmount * (1.0 - baseLuminance * 0.5));
  
  frag = vec4(finalResult, 1.0);
}`;

// Advanced render with professional tone mapping
const FS_RENDER = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_dye;
uniform vec2 u_texelSize;
uniform float u_brightness;
uniform float u_contrast;
uniform vec3 u_colorBalance;
uniform float u_exposure;
uniform float u_whitePoint;
uniform int u_tonemapMode; // 0=Reinhard, 1=ACES, 2=Filmic

// Reinhard tone mapping
vec3 reinhard(vec3 color, float whitePoint) {
    return color / (1.0 + color / (whitePoint * whitePoint));
}

// Simplified ACES tone mapping
vec3 aces(vec3 color) {
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0);
}

// Filmic tone mapping
vec3 filmic(vec3 color) {
    vec3 x = max(vec3(0.0), color - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

// Local contrast enhancement
vec3 localContrast(sampler2D tex, vec2 uv, vec2 texelSize, float strength) {
    vec3 center = texture(tex, uv).rgb;
    vec3 blur = texture(tex, uv + vec2(-1, -1) * texelSize).rgb;
    blur += texture(tex, uv + vec2(0, -1) * texelSize).rgb;
    blur += texture(tex, uv + vec2(1, -1) * texelSize).rgb;
    blur += texture(tex, uv + vec2(-1, 0) * texelSize).rgb;
    blur += texture(tex, uv + vec2(1, 0) * texelSize).rgb;
    blur += texture(tex, uv + vec2(-1, 1) * texelSize).rgb;
    blur += texture(tex, uv + vec2(0, 1) * texelSize).rgb;
    blur += texture(tex, uv + vec2(1, 1) * texelSize).rgb;
    blur /= 8.0;
    
    return center + (center - blur) * strength;
}

void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec3 c = texture(u_dye, uv).rgb;
  
  // Local contrast enhancement for detail preservation
  c = localContrast(u_dye, uv, u_texelSize, 0.3);
  
  // Color balance adjustment
  c *= u_colorBalance;
  
  // Exposure adjustment (before tone mapping)
  c *= pow(2.0, u_exposure);
  
  // Brightness and contrast
  c = (c - 0.5) * u_contrast + 0.5 + u_brightness;
  
  // Advanced tone mapping
  if (u_tonemapMode == 1) {
      c = aces(c);
  } else if (u_tonemapMode == 2) {
      c = filmic(c);
  } else {
      c = reinhard(c, u_whitePoint);
  }
  
  // Gamma correction
  c = pow(max(c, vec3(0.0)), vec3(1.0/2.2));
  
  frag = vec4(c, 1.0);
}`;

function uniformLoc(gl, prog, name) {
  const loc = gl.getUniformLocation(prog, name);
  if (loc === null) {
    console.warn(`Uniform not found: ${name} (this is OK if unused by shader)`);
    return null;
  }
  return loc;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Safe uniform setting functions
function setUniform1f(gl, location, value) {
  if (location !== null) gl.uniform1f(location, value);
}

function setUniform1i(gl, location, value) {
  if (location !== null) gl.uniform1i(location, value);
}

function setUniform2f(gl, location, x, y) {
  if (location !== null) gl.uniform2f(location, x, y);
}

function setUniform3f(gl, location, x, y, z) {
  if (location !== null) gl.uniform3f(location, x, y, z);
}

export class FluidSim {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    const gl = getGL(canvas);
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;

    if (!checkFloatSupport(gl)) {
      throw new Error("Required float framebuffer support missing (EXT_color_buffer_float)");
    }

    // Settings
    this.settings = {
      dyeDissipation: 0.995,
      velDissipation: 0.998,
      pressureIters: 18,
      vorticity: 1.0,
      splatRadius: 0.015,
      qualityScale: 0.75, // 0..1, sim resolution scalar
      maxDPR: 1.8,
      // Visual effects
      bloomEnabled: true,
      bloomThreshold: 0.6,
      bloomKnee: 0.7,
      bloomIntensity: 0.8,
      bloomSaturation: 1.2,
      bloomBlendMode: 0, // Start with additive (safest)
      detailPreservation: 0.2, // Reduced for stability
      baseTexture: 0.15, // Always preserve some texture detail
      brightness: 0.0,
      contrast: 1.0,
      colorBalance: [1.0, 1.0, 1.0],
      // Advanced rendering
      exposure: 0.0,
      whitePoint: 2.0,
      tonemapMode: 0, // Start with Reinhard (safest)
      // Color palettes
      colorPalette: 'rainbow',
      paletteSpeed: 1.0,
    };
    Object.assign(this.settings, options);

    this._initGL();
    this._resize();
  }

  _initGL() {
    const gl = this.gl;
    this.vao = createFullscreenVAO(gl);
    this.programs = {
      advect: createProgram(gl, VS_SCREEN, FS_ADVECT),
      divergence: createProgram(gl, VS_SCREEN, FS_DIVERGENCE),
      pressure: createProgram(gl, VS_SCREEN, FS_PRESSURE),
      gradient: createProgram(gl, VS_SCREEN, FS_GRADIENT),
      curl: createProgram(gl, VS_SCREEN, FS_CURL),
      vorticity: createProgram(gl, VS_SCREEN, FS_VORTICITY),
      splat: createProgram(gl, VS_SCREEN, FS_SPLAT),
      render: createProgram(gl, VS_SCREEN, FS_RENDER),
      bloomPrefilter: createProgram(gl, VS_SCREEN, FS_BLOOM_PREFILTER),
      blur: createProgram(gl, VS_SCREEN, FS_BLUR),
      bloomFinal: createProgram(gl, VS_SCREEN, FS_BLOOM_FINAL),
    };

    // Uniform locations cached
    const U = (prog, name) => uniformLoc(gl, prog, name);
    this.u = {
      advect: {
        u_source: U(this.programs.advect, "u_source"),
        u_velocity: U(this.programs.advect, "u_velocity"),
        u_texelSize: U(this.programs.advect, "u_texelSize"),
        u_resolution: U(this.programs.advect, "u_resolution"),
        u_dt: U(this.programs.advect, "u_dt"),
        u_dissipation: U(this.programs.advect, "u_dissipation"),
      },
      divergence: {
        u_velocity: U(this.programs.divergence, "u_velocity"),
        u_texelSize: U(this.programs.divergence, "u_texelSize"),
      },
      pressure: {
        u_pressure: U(this.programs.pressure, "u_pressure"),
        u_divergence: U(this.programs.pressure, "u_divergence"),
        u_texelSize: U(this.programs.pressure, "u_texelSize"),
      },
      gradient: {
        u_velocity: U(this.programs.gradient, "u_velocity"),
        u_pressure: U(this.programs.gradient, "u_pressure"),
        u_texelSize: U(this.programs.gradient, "u_texelSize"),
      },
      curl: {
        u_velocity: U(this.programs.curl, "u_velocity"),
        u_texelSize: U(this.programs.curl, "u_texelSize"),
      },
      vorticity: {
        u_velocity: U(this.programs.vorticity, "u_velocity"),
        u_curl: U(this.programs.vorticity, "u_curl"),
        u_texelSize: U(this.programs.vorticity, "u_texelSize"),
        u_dt: U(this.programs.vorticity, "u_dt"),
        u_eps: U(this.programs.vorticity, "u_eps"),
      },
      splat: {
        u_target: U(this.programs.splat, "u_target"),
        u_point: U(this.programs.splat, "u_point"),
        u_value: U(this.programs.splat, "u_value"),
        u_radius: U(this.programs.splat, "u_radius"),
        u_texelSize: U(this.programs.splat, "u_texelSize"),
      },
      render: {
        u_dye: U(this.programs.render, "u_dye"),
        u_texelSize: U(this.programs.render, "u_texelSize"),
        u_brightness: U(this.programs.render, "u_brightness"),
        u_contrast: U(this.programs.render, "u_contrast"),
        u_colorBalance: U(this.programs.render, "u_colorBalance"),
        u_exposure: U(this.programs.render, "u_exposure"),
        u_whitePoint: U(this.programs.render, "u_whitePoint"),
        u_tonemapMode: U(this.programs.render, "u_tonemapMode"),
      },
      bloomPrefilter: {
        u_dye: U(this.programs.bloomPrefilter, "u_dye"),
        u_texelSize: U(this.programs.bloomPrefilter, "u_texelSize"),
        u_threshold: U(this.programs.bloomPrefilter, "u_threshold"),
        u_knee: U(this.programs.bloomPrefilter, "u_knee"),
      },
      blur: {
        u_source: U(this.programs.blur, "u_source"),
        u_texelSize: U(this.programs.blur, "u_texelSize"),
        u_direction: U(this.programs.blur, "u_direction"),
      },
      bloomFinal: {
        u_base: U(this.programs.bloomFinal, "u_base"),
        u_bloom: U(this.programs.bloomFinal, "u_bloom"),
        u_texelSize: U(this.programs.bloomFinal, "u_texelSize"),
        u_intensity: U(this.programs.bloomFinal, "u_intensity"),
        u_saturation: U(this.programs.bloomFinal, "u_saturation"),
        u_blendMode: U(this.programs.bloomFinal, "u_blendMode"),
        u_detailPreservation: U(this.programs.bloomFinal, "u_detailPreservation"),
        u_baseTexture: U(this.programs.bloomFinal, "u_baseTexture"),
        u_brightness: U(this.programs.bloomFinal, "u_brightness"),
        u_contrast: U(this.programs.bloomFinal, "u_contrast"),
        u_colorBalance: U(this.programs.bloomFinal, "u_colorBalance"),
        u_exposure: U(this.programs.bloomFinal, "u_exposure"),
        u_whitePoint: U(this.programs.bloomFinal, "u_whitePoint"),
        u_tonemapMode: U(this.programs.bloomFinal, "u_tonemapMode"),
      },
    };

    // Geometry
    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
  }

  _allocate(width, height) {
    const gl = this.gl;
    const filter = gl.NEAREST; // manual bilerp in shader guarantees quality, avoids extension pitfalls
    const type = gl.HALF_FLOAT;
    // Choose sized internal formats
    const RG = gl.RG16F, RG_FMT = gl.RG, R = gl.R16F, R_FMT = gl.RED, RGBA = gl.RGBA16F, RGBA_FMT = gl.RGBA;

    // Velocity (RG), Pressure (R), Divergence (R), Curl (R), Dye (RGBA)
    this.velocity = makeDoubleFBO(gl, width, height, RG, RG_FMT, type, filter);
    this.pressure = makeDoubleFBO(gl, width, height, R, R_FMT, type, filter);
    this.divergence = makeFBO(gl, width, height, R, R_FMT, type, filter);
    this.curl = makeFBO(gl, width, height, R, R_FMT, type, filter);
    this.dye = makeDoubleFBO(gl, width, height, RGBA, RGBA_FMT, type, filter);
    
    // Bloom framebuffers (full resolution to avoid sampling issues)
    const bloomW = width;
    const bloomH = height;
    this.bloom = {
      prefilter: makeFBO(gl, bloomW, bloomH, RGBA, RGBA_FMT, type, filter),
      blur: makeDoubleFBO(gl, bloomW, bloomH, RGBA, RGBA_FMT, type, filter),
    };
  }

  _dispose() {
    this.velocity?.dispose?.();
    this.pressure?.dispose?.();
    const gl = this.gl;
    if (this.divergence) { gl.deleteFramebuffer(this.divergence.fb); gl.deleteTexture(this.divergence.tex); }
    if (this.curl) { gl.deleteFramebuffer(this.curl.fb); gl.deleteTexture(this.curl.tex); }
    if (this.dye) this.dye.dispose();
    if (this.bloom) {
      if (this.bloom.prefilter) { gl.deleteFramebuffer(this.bloom.prefilter.fb); gl.deleteTexture(this.bloom.prefilter.tex); }
      if (this.bloom.blur) this.bloom.blur.dispose();
    }
  }

  _resize() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, this.settings.maxDPR);
    const scale = clamp(this.settings.qualityScale, 0.25, 1.0);
    const width = Math.max(16, Math.floor(this.canvas.clientWidth * dpr * scale));
    const height = Math.max(16, Math.floor(this.canvas.clientHeight * dpr * scale));
    if (gl.drawingBufferWidth === width && gl.drawingBufferHeight === height && this._w === width && this._h === height) {
      return;
    }
    this.canvas.width = Math.max(2, Math.floor(this.canvas.clientWidth * dpr));
    this.canvas.height = Math.max(2, Math.floor(this.canvas.clientHeight * dpr));
    this._dispose();
    this._allocate(width, height);
    this._w = width; this._h = height;
  }

  reset() {
    this._dispose();
    this._allocate(this._w, this._h);
  }

  _setCommonUniforms(program, texelW, texelH) {
    const gl = this.gl;
    setUniform2f(gl, this.u[program].u_texelSize, 1 / texelW, 1 / texelH);
  }

  step(dt) {
    const gl = this.gl;
    const w = this.velocity.read.w, h = this.velocity.read.h;
    const texel = [1 / w, 1 / h];
    gl.bindVertexArray(this.vao);

    // Curl
    bindFBO(gl, this.curl);
    setViewport(gl, this.curl);
    gl.useProgram(this.programs.curl);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    setUniform1i(gl, this.u.curl.u_velocity, 0);
    setUniform2f(gl, this.u.curl.u_texelSize, texel[0], texel[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Vorticity confinement
    if (this.settings.vorticity > 0.0) {
      bindFBO(gl, this.velocity.write);
      setViewport(gl, this.velocity.write);
      gl.useProgram(this.programs.vorticity);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
      setUniform1i(gl, this.u.vorticity.u_velocity, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.curl.tex);
      setUniform1i(gl, this.u.vorticity.u_curl, 1);
      setUniform2f(gl, this.u.vorticity.u_texelSize, texel[0], texel[1]);
      setUniform1f(gl, this.u.vorticity.u_dt, dt);
      setUniform1f(gl, this.u.vorticity.u_eps, this.settings.vorticity);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.velocity.swap();
    }

    // Advect velocity
    bindFBO(gl, this.velocity.write);
    setViewport(gl, this.velocity.write);
    gl.useProgram(this.programs.advect);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    setUniform1i(gl, this.u.advect.u_source, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    setUniform1i(gl, this.u.advect.u_velocity, 1);
    setUniform2f(gl, this.u.advect.u_texelSize, texel[0], texel[1]);
    setUniform2f(gl, this.u.advect.u_resolution, w, h);
    setUniform1f(gl, this.u.advect.u_dt, dt);
    setUniform1f(gl, this.u.advect.u_dissipation, this.settings.velDissipation);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.velocity.swap();

    // Divergence
    bindFBO(gl, this.divergence);
    setViewport(gl, this.divergence);
    gl.useProgram(this.programs.divergence);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    setUniform1i(gl, this.u.divergence.u_velocity, 0);
    setUniform2f(gl, this.u.divergence.u_texelSize, texel[0], texel[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pressure solve iterations
    gl.useProgram(this.programs.pressure);
    for (let i = 0; i < this.settings.pressureIters; i++) {
      bindFBO(gl, this.pressure.write);
      setViewport(gl, this.pressure.write);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.tex);
      setUniform1i(gl, this.u.pressure.u_pressure, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.divergence.tex);
      setUniform1i(gl, this.u.pressure.u_divergence, 1);
      setUniform2f(gl, this.u.pressure.u_texelSize, texel[0], texel[1]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.pressure.swap();
    }

    // Subtract gradient
    bindFBO(gl, this.velocity.write);
    setViewport(gl, this.velocity.write);
    gl.useProgram(this.programs.gradient);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    setUniform1i(gl, this.u.gradient.u_velocity, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.tex);
    setUniform1i(gl, this.u.gradient.u_pressure, 1);
    setUniform2f(gl, this.u.gradient.u_texelSize, texel[0], texel[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.velocity.swap();

    // Advect dye
    bindFBO(gl, this.dye.write);
    setViewport(gl, this.dye.write);
    gl.useProgram(this.programs.advect);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.dye.read.tex);
    setUniform1i(gl, this.u.advect.u_source, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    setUniform1i(gl, this.u.advect.u_velocity, 1);
    setUniform2f(gl, this.u.advect.u_texelSize, texel[0], texel[1]);
    setUniform2f(gl, this.u.advect.u_resolution, w, h);
    setUniform1f(gl, this.u.advect.u_dt, dt);
    setUniform1f(gl, this.u.advect.u_dissipation, this.settings.dyeDissipation);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.dye.swap();
  }

  renderToScreen() {
    const gl = this.gl;
    
    if (this.settings.bloomEnabled) {
      // Bloom prefilter - extract bright areas
      bindFBO(gl, this.bloom.prefilter);
      setViewport(gl, this.bloom.prefilter);
      gl.useProgram(this.programs.bloomPrefilter);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.dye.read.tex);
      setUniform1i(gl, this.u.bloomPrefilter.u_dye, 0);
      setUniform2f(gl, this.u.bloomPrefilter.u_texelSize, 1 / this.dye.read.w, 1 / this.dye.read.h);
      setUniform1f(gl, this.u.bloomPrefilter.u_threshold, this.settings.bloomThreshold);
      setUniform1f(gl, this.u.bloomPrefilter.u_knee, this.settings.bloomKnee);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      
      // Horizontal blur
      bindFBO(gl, this.bloom.blur.write);
      setViewport(gl, this.bloom.blur.write);
      gl.useProgram(this.programs.blur);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.bloom.prefilter.tex);
      setUniform1i(gl, this.u.blur.u_source, 0);
      setUniform2f(gl, this.u.blur.u_texelSize, 1 / this.bloom.prefilter.w, 1 / this.bloom.prefilter.h);
      setUniform2f(gl, this.u.blur.u_direction, 1, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.bloom.blur.swap();
      
      // Vertical blur
      bindFBO(gl, this.bloom.blur.write);
      setViewport(gl, this.bloom.blur.write);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.bloom.blur.read.tex);
      setUniform2f(gl, this.u.blur.u_direction, 0, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.bloom.blur.swap();
      
      // Final composite with bloom
      bindFBO(gl, null);
      setViewport(gl, null);
      gl.useProgram(this.programs.bloomFinal);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.dye.read.tex);
      setUniform1i(gl, this.u.bloomFinal.u_base, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.bloom.blur.read.tex);
      setUniform1i(gl, this.u.bloomFinal.u_bloom, 1);
      setUniform2f(gl, this.u.bloomFinal.u_texelSize, 1 / gl.drawingBufferWidth, 1 / gl.drawingBufferHeight);
      setUniform1f(gl, this.u.bloomFinal.u_intensity, this.settings.bloomIntensity);
      setUniform1f(gl, this.u.bloomFinal.u_saturation, this.settings.bloomSaturation);
      setUniform1i(gl, this.u.bloomFinal.u_blendMode, this.settings.bloomBlendMode);
      setUniform1f(gl, this.u.bloomFinal.u_detailPreservation, this.settings.detailPreservation);
      setUniform1f(gl, this.u.bloomFinal.u_baseTexture, this.settings.baseTexture);
      setUniform1f(gl, this.u.bloomFinal.u_brightness, this.settings.brightness);
      setUniform1f(gl, this.u.bloomFinal.u_contrast, this.settings.contrast);
      setUniform3f(gl, this.u.bloomFinal.u_colorBalance, ...this.settings.colorBalance);
      setUniform1f(gl, this.u.bloomFinal.u_exposure, this.settings.exposure);
      setUniform1f(gl, this.u.bloomFinal.u_whitePoint, this.settings.whitePoint);
      setUniform1i(gl, this.u.bloomFinal.u_tonemapMode, this.settings.tonemapMode);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } else {
      // Simple render without bloom
      bindFBO(gl, null);
      setViewport(gl, null);
      gl.useProgram(this.programs.render);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.dye.read.tex);
      setUniform1i(gl, this.u.render.u_dye, 0);
      setUniform2f(gl, this.u.render.u_texelSize, 1 / this.dye.read.w, 1 / this.dye.read.h);
      setUniform1f(gl, this.u.render.u_brightness, this.settings.brightness);
      setUniform1f(gl, this.u.render.u_contrast, this.settings.contrast);
      setUniform3f(gl, this.u.render.u_colorBalance, ...this.settings.colorBalance);
      setUniform1f(gl, this.u.render.u_exposure, this.settings.exposure);
      setUniform1f(gl, this.u.render.u_whitePoint, this.settings.whitePoint);
      setUniform1i(gl, this.u.render.u_tonemapMode, this.settings.tonemapMode);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  splat(point01, value3, radius01, target) {
    const gl = this.gl;
    const tgt = target === "velocity" ? this.velocity : this.dye;
    bindFBO(gl, tgt.write);
    setViewport(gl, tgt.write);
    gl.useProgram(this.programs.splat);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tgt.read.tex);
    setUniform1i(gl, this.u.splat.u_target, 0);
    setUniform2f(gl, this.u.splat.u_point, point01[0], point01[1]);
    setUniform3f(gl, this.u.splat.u_value, value3[0], value3[1], value3[2] ?? 0.0);
    setUniform1f(gl, this.u.splat.u_radius, radius01);
    setUniform2f(gl, this.u.splat.u_texelSize, 1 / tgt.read.w, 1 / tgt.read.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    tgt.swap();
  }
}
