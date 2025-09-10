"use strict";

// Minimal WebGL2 helpers focused on safety and clarity.

export function getGL(canvas) {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    desynchronized: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: false,
  });
  return gl;
}

export function createProgram(gl, vsSource, fsSource) {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  if (!vs || !fs) throw new Error("Failed to create shaders");

  gl.shaderSource(vs, vsSource);
  gl.shaderSource(fs, fsSource);
  gl.compileShader(vs);
  gl.compileShader(fs);

  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(vs) || "";
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error("Vertex shader error: " + info);
  }
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(fs) || "";
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error("Fragment shader error: " + info);
  }

  const prog = gl.createProgram();
  if (!prog) throw new Error("Failed to create program");
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) || "";
    gl.deleteProgram(prog);
    throw new Error("Program link error: " + info);
  }
  return prog;
}

export function createFullscreenVAO(gl) {
  const vao = gl.createVertexArray();
  if (!vao) throw new Error("Failed to create VAO");
  gl.bindVertexArray(vao);
  // Full-screen triangle
  const buf = gl.createBuffer();
  if (!buf) throw new Error("Failed to create buffer");
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  // positions for a single big triangle covering the viewport
  const data = new Float32Array([
    -1, -1,
    3, -1,
    -1, 3,
  ]);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const loc = 0; // a_pos location
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}

export function checkFloatSupport(gl) {
  // Require rendering to float color attachments.
  const ext = gl.getExtension("EXT_color_buffer_float");
  return !!ext;
}

export function makeFBO(gl, w, h, internalFormat, format, type, filter) {
  const tex = gl.createTexture();
  const fb = gl.createFramebuffer();
  if (!tex || !fb) throw new Error("Failed to create FBO");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(fb);
    gl.deleteTexture(tex);
    throw new Error("Incomplete FBO: " + status);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { tex, fb, w, h };
}

export function makeDoubleFBO(gl, w, h, internalFormat, format, type, filter) {
  const a = makeFBO(gl, w, h, internalFormat, format, type, filter);
  const b = makeFBO(gl, w, h, internalFormat, format, type, filter);
  return {
    get read() { return a; },
    get write() { return b; },
    swap() {
      const t = a.tex; const tf = a.fb; const tw = a.w; const th = a.h;
      a.tex = b.tex; a.fb = b.fb; a.w = b.w; a.h = b.h;
      b.tex = t; b.fb = tf; b.w = tw; b.h = th;
    },
    dispose() {
      gl.deleteFramebuffer(a.fb);
      gl.deleteTexture(a.tex);
      gl.deleteFramebuffer(b.fb);
      gl.deleteTexture(b.tex);
    },
  };
}

export function setViewport(gl, target) {
  const w = target?.w ?? gl.drawingBufferWidth;
  const h = target?.h ?? gl.drawingBufferHeight;
  gl.viewport(0, 0, w, h);
}

export function bindFBO(gl, target) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null);
}

