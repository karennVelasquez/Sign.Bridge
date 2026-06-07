/* ═══════════════════════════════════════════
   Sign.Bridge — app.js
═══════════════════════════════════════════ */

const WS_URL  = 'ws://localhost:8000/ws/predict';
const API_URL = 'http://localhost:8000';
const FPS_MS  = 120;
const COL_MS  = 100;

// Canvas para predicción WS (compartido)
const _canvas = document.createElement('canvas');
_canvas.width = _canvas.height = 320;
const _ctx = _canvas.getContext('2d');

// Canvas separado para recolección de muestras (evita race condition con WS)
// Resolución mayor → MediaPipe detecta mejor las manos
const _collectCanvas = document.createElement('canvas');
_collectCanvas.width  = 640;
_collectCanvas.height = 480;
const _collectCtx = _collectCanvas.getContext('2d');

const SIGNS = {
  A:'👊', B:'✋', C:'🤏', D:'☝️', E:'🤞', F:'👌', G:'👉', H:'🤙',
  I:'🤌', J:'✍️', K:'✌️', L:'🤙', M:'✊', N:'🤛', Ñ:'🤜', O:'⭕',
  P:'👇', Q:'👆', R:'🤞', S:'✊', T:'👍', U:'🤟', V:'✌️', W:'🖖',
  X:'☝️', Y:'🤙', Z:'☝️'
};

const State = {
  ws: null,
  captureStream: null,
  predictStream: null,
  wsInterval: null,
  collectInterval: null,
  timerInterval: null,
  camOn: false,
  predCamOn: false,
  currentWord: '',
  targetSamples: 200,
  capturedCount: 0,
  isRecording: false,
  predHistory: [],
  lastStableLetter: '',
  lastStableTime: 0,
  STABLE_COOLDOWN: 2000,
  detectedCount: 0,
  confSum: 0,
  seconds: 0,
  demoIdx: 0,
};

/* ═══════════════════════════════════════════
   NAVEGACIÓN
═══════════════════════════════════════════ */
function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + id).classList.add('active');
  const isApp = id === 'app';
  document.getElementById('nav-app-tabs').style.display     = isApp ? 'flex' : 'none';
  document.getElementById('nav-landing-tabs').style.display = isApp ? 'none' : 'flex';
  document.getElementById('conn-status').style.display      = isApp ? 'flex' : 'none';
  if (isApp) window.scrollTo(0, 0);
}

function showAppPanel(id, el) {
  document.querySelectorAll('.app-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('#nav-app-tabs .n-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  if (el) {
    el.classList.add('active');
  } else {
    const map = { capture: 0, train: 1, predict: 2, dictionary: 3 };
    const tabs = document.querySelectorAll('#nav-app-tabs .n-tab');
    if (tabs[map[id]]) tabs[map[id]].classList.add('active');
  }
}

function scrollAnchor(hash) {
  const el = document.querySelector(hash);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ═══════════════════════════════════════════
   WEBSOCKET — predicción
═══════════════════════════════════════════ */
function connectWS(onOpen) {
  // Si ya está abierto, reutilizar
  if (State.ws && State.ws.readyState === WebSocket.OPEN) {
    if (onOpen) onOpen();
    return;
  }
  // Cerrar cualquier WS anterior
  if (State.ws) {
    State.ws.onclose = null;
    State.ws.close();
    State.ws = null;
  }

  State.ws = new WebSocket(WS_URL);

  State.ws.onopen = () => {
    setConnStatus(true);
    document.getElementById('conn-error').classList.remove('visible');
    showToast('Conectado al backend', 'success');
    if (onOpen) onOpen();
  };
  State.ws.onmessage = e => handlePrediction(JSON.parse(e.data));
  State.ws.onerror   = () => {
    setConnStatus(false);
    // Solo mostrar el banner de error si NO estamos en modo captura activa
    if (!State.camOn) {
      document.getElementById('conn-error').classList.add('visible');
    }
    // No mostrar toast de error si la cámara de captura está activa
    // (el WS es opcional durante captura, las muestras se guardan igual)
    if (!State.camOn) {
      showToast('No se pudo conectar. ¿Está corriendo el servidor?', 'error');
    }
  };
  State.ws.onclose = () => {
    setConnStatus(false);
    clearInterval(State.wsInterval);
  };
}

function setConnStatus(on) {
  const pill = document.getElementById('conn-status');
  const dot  = document.getElementById('conn-dot');
  const txt  = document.getElementById('conn-text');
  pill.className  = 'status-pill ' + (on ? 'on' : 'off');
  dot.classList.toggle('pulse', on);
  txt.textContent = on ? 'Conectado' : 'Desconectado';
}

/* ═══════════════════════════════════════════
   PANEL CAPTURA
═══════════════════════════════════════════ */
function setupCapture() {
  const wordInput    = document.getElementById('input-sign-word');
  const samplesInput = document.getElementById('input-sign-samples');
  const startBtn     = document.getElementById('btn-start-capture');
  if (!wordInput || !startBtn) return;

  wordInput.addEventListener('input', () => {
    wordInput.value = wordInput.value.toUpperCase().replace(/\s+/g, '_');
  });

  startBtn.addEventListener('click', () => {
    const word = wordInput.value.trim();
    if (!word) { showToast('Escribe la palabra o seña primero', 'error'); wordInput.focus(); return; }
    beginCapture(word, parseInt(samplesInput.value) || 200);
  });

  document.getElementById('btn-cancel-capture')?.addEventListener('click', cancelCapture);
  document.getElementById('btn-toggle-rec')?.addEventListener('click', toggleRecording);
  document.getElementById('btn-cam-cap')?.addEventListener('click', () => {
    if (State.camOn) stopCaptureCamera(); else startCaptureCamera();
  });
}

function beginCapture(word, samples) {
  State.currentWord   = word;
  State.targetSamples = samples;
  State.capturedCount = 0;
  State.isRecording   = false;

  document.getElementById('asb-word').textContent           = word;
  document.getElementById('asb-samples-target').textContent = samples;
  document.getElementById('asb-samples-done').textContent   = '0';
  document.getElementById('active-sign-bar').classList.add('visible');
  updateCaptureProgress(0, samples);
  document.getElementById('capture-setup-form').style.display  = 'none';
  document.getElementById('capture-cam-section').style.display = 'block';
  startCaptureCamera();
}

function cancelCapture() {
  stopCollectLoop();
  stopCaptureCamera();
  document.getElementById('active-sign-bar').classList.remove('visible');
  document.getElementById('capture-setup-form').style.display  = '';
  document.getElementById('capture-cam-section').style.display = 'none';
  document.getElementById('input-sign-word').value = '';
  State.currentWord = ''; State.isRecording = false; State.capturedCount = 0;
}

function toggleRecording() {
  if (!State.camOn) { showToast('Activa la cámara primero', 'error'); return; }
  State.isRecording = !State.isRecording;
  const btn = document.getElementById('btn-toggle-rec');
  const rec = document.getElementById('rec-overlay');
  const lbl = document.getElementById('cam-label-cap');
  if (State.isRecording) {
    if (btn) { btn.innerHTML = '⏸ Pausar'; btn.className = 'btn btn-danger'; }
    rec?.classList.add('visible');
    if (lbl) { lbl.textContent = '● Grabando muestras…'; lbl.classList.add('visible'); }
    showToast('Grabando muestras de "' + State.currentWord + '"', 'info');
    startCollectLoop();
  } else {
    if (btn) { btn.innerHTML = '▶ Grabar'; btn.className = 'btn btn-success'; }
    rec?.classList.remove('visible');
    if (lbl) lbl.textContent = 'Pausado';
    showToast('Pausado — ' + State.capturedCount + ' muestras guardadas', 'info');
    stopCollectLoop();
  }
}

function updateCaptureProgress(done, total) {
  const capped = Math.min(done, total);
  const pct    = total > 0 ? (capped / total) * 100 : 0;
  const fill   = document.getElementById('cp-bar-fill');
  const meta   = document.getElementById('cp-count');
  const pctEl  = document.getElementById('cp-pct');
  if (fill)  fill.style.width  = pct + '%';
  if (meta)  meta.textContent  = capped + ' / ' + total + ' muestras';
  if (pctEl) pctEl.textContent = Math.round(pct) + '%';
  const asbDone = document.getElementById('asb-samples-done');
  if (asbDone) asbDone.textContent = capped;

  if (capped >= total && total > 0) {
    stopCollectLoop();
    State.isRecording = false; State.capturedCount = total;
    const btn = document.getElementById('btn-toggle-rec');
    if (btn) { btn.innerHTML = '✓ Completado'; btn.className = 'btn btn-secondary'; btn.disabled = true; }
    document.getElementById('rec-overlay')?.classList.remove('visible');
    const lbl = document.getElementById('cam-label-cap');
    if (lbl) lbl.textContent = `✓ ${total} muestras guardadas`;
    showToast(`✓ ${total} muestras de "${State.currentWord}" completadas`, 'success');
    setTimeout(loadTrainedWords, 800);
  }
}

function startCollectLoop() {
  stopCollectLoop();
  const video = document.getElementById('video-capture');
  let inFlight = false;

  State.collectInterval = setInterval(async () => {
    if (!State.isRecording) return;
    if (State.capturedCount >= State.targetSamples) { stopCollectLoop(); return; }
    if (inFlight) return;
    if (!video) { console.warn('[collect] sin elemento video'); return; }
    if (video.readyState < 2) { console.warn('[collect] video.readyState =', video.readyState); return; }
    if (!video.videoWidth)    { console.warn('[collect] video.videoWidth = 0'); return; }
    inFlight = true;

    try {
      // Dibujar el video al canvas manteniendo proporciones reales del video
      const vw = video.videoWidth, vh = video.videoHeight;
      if (_collectCanvas.width !== vw || _collectCanvas.height !== vh) {
        _collectCanvas.width  = vw;
        _collectCanvas.height = vh;
        console.log('[collect] canvas redimensionado a', vw, 'x', vh);
      }
      _collectCtx.drawImage(video, 0, 0, vw, vh);
      const blob = await new Promise(r => _collectCanvas.toBlob(r, 'image/jpeg', 0.9));
      if (!blob) {
        console.warn('[collect] toBlob retornó null');
        inFlight = false;
        return;
      }
      // Log solo cada 10 frames para no llenar consola
      if (!window._collectFrameN) window._collectFrameN = 0;
      window._collectFrameN++;
      if (window._collectFrameN % 10 === 1) {
        console.log('[collect] enviando frame', window._collectFrameN, '- blob:', blob.size, 'bytes, video:', vw + 'x' + vh);
      }

      const res  = await fetch(`${API_URL}/api/sample`, {
        method: 'POST',
        headers: {
          'X-Word':   State.currentWord,
          'X-Target': String(State.targetSamples),
        },
        body: blob,
      });

      if (!res.ok) {
        console.warn('[collect] HTTP error:', res.status, res.statusText);
        const txt = await res.text();
        console.warn('[collect] body:', txt);
        inFlight = false;
        return;
      }
      const data = await res.json();
      console.log('[collect]', data);   // ← log every response for debugging

      if (data.saved) {
        State.capturedCount = data.total;
        updateCaptureProgress(data.total, State.targetSamples);
        const lbl = document.getElementById('cam-label-cap');
        if (lbl) lbl.textContent = `● Grabando (${data.total}/${State.targetSamples})`;
      } else if (data.reason === 'target_reached') {
        State.capturedCount = data.total;
        updateCaptureProgress(data.total, State.targetSamples);
        stopCollectLoop();
      } else if (data.reason === 'throttle') {
        // ignorar - se reintentará en el próximo tick
      } else if (data.hand_detected === false) {
        const lbl = document.getElementById('cam-label-cap');
        if (lbl && State.isRecording) {
          lbl.textContent = '✋ Pon la mano frente a la cámara';
          lbl.classList.add('visible');
        }
      } else if (data.error) {
        console.error('[collect] server error:', data.error);
        showToast('Error: ' + data.error, 'error');
      }
    } catch (err) {
      console.error('[collect] exception:', err);
    } finally {
      inFlight = false;
    }
  }, COL_MS);
}

function stopCollectLoop() {
  clearInterval(State.collectInterval);
  State.collectInterval = null;
}

/* ═══════════════════════════════════════════
   CÁMARA — CAPTURA
═══════════════════════════════════════════ */
async function startCaptureCamera() {
  try {
    State.captureStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    const v = document.getElementById('video-capture');
    v.srcObject = State.captureStream; v.classList.add('active'); await v.play();
    State.camOn = true;
    _setCamUI('cap', true);
    startTimer();
    // WS para preview de detección es opcional — no bloquea la captura de muestras
    try { connectWS(() => startWsLoop('cap')); } catch (_) {}
  } catch (err) {
    showToast('Cámara no disponible: ' + err.message, 'error');
  }
}

function stopCaptureCamera() {
  State.captureStream?.getTracks().forEach(t => t.stop());
  stopCollectLoop();
  clearInterval(State.wsInterval);
  stopTimer();
  State.camOn = false; State.isRecording = false;
  const v = document.getElementById('video-capture');
  if (v) { v.srcObject = null; v.classList.remove('active'); }
  _setCamUI('cap', false);
  document.getElementById('rec-overlay')?.classList.remove('visible');
}

/* ═══════════════════════════════════════════
   CÁMARA — PREDICCIÓN
   WS independiente para no interferir con captura
═══════════════════════════════════════════ */
async function togglePredCamera() {
  State.predCamOn ? stopPredCamera() : await startPredCamera();
}

async function startPredCamera() {
  try {
    State.predictStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    const v = document.getElementById('video-predict');
    v.srcObject = State.predictStream; v.classList.add('active'); await v.play();
    State.predCamOn = true;
    _setCamUI('pred', true);
    State.predHistory = []; State.lastStableLetter = ''; State.lastStableTime = 0;
    _renderPredWord();
    // WS propio para predicción — no comparte con captura
    connectWS(() => startWsLoop('pred'));
  } catch (err) {
    showToast('Cámara no disponible: ' + err.message, 'error');
  }
}

function stopPredCamera() {
  State.predictStream?.getTracks().forEach(t => t.stop());
  clearInterval(State.wsInterval);
  State.predCamOn = false;
  const v = document.getElementById('video-predict');
  if (v) { v.srcObject = null; v.classList.remove('active'); }
  _setCamUI('pred', false);
}

function startWsLoop(mode) {
  clearInterval(State.wsInterval);
  const videoId = mode === 'cap' ? 'video-capture' : 'video-predict';
  State.wsInterval = setInterval(() => {
    // No enviar frames al WS mientras se está grabando muestras
    // (evita saturar el extractor de MediaPipe en el servidor)
    if (mode === 'cap' && State.isRecording) return;
    const v = document.getElementById(videoId);
    if (!v?.videoWidth) return;
    if (!State.ws || State.ws.readyState !== WebSocket.OPEN) return;
    _ctx.drawImage(v, 0, 0, _canvas.width, _canvas.height);
    _canvas.toBlob(b => {
      if (b && State.ws?.readyState === WebSocket.OPEN) State.ws.send(b);
    }, 'image/jpeg', 0.85);
  }, FPS_MS);
}

function _setCamUI(mode, on) {
  const map = {
    cap:  { ph:'cam-ph-cap',  box:'hbox-cap',  lbl:'cam-label-cap',  badge:'cam-badge-cap',  btn:'btn-cam-cap' },
    pred: { ph:'cam-ph-pred', box:'hbox-pred', lbl:'cam-label-pred', badge:'cam-badge-pred', btn:'btn-cam-pred' },
  };
  const ids = map[mode]; if (!ids) return;
  const ph    = document.getElementById(ids.ph);
  const box   = document.getElementById(ids.box);
  const lbl   = document.getElementById(ids.lbl);
  const badge = document.getElementById(ids.badge);
  const btn   = document.getElementById(ids.btn);
  if (ph)    ph.style.display = on ? 'none' : '';
  if (box)   box.classList.toggle('visible', on);
  if (lbl)   { lbl.classList.toggle('visible', on); if (on) lbl.textContent = 'Detectando mano…'; }
  if (badge) { badge.textContent = on ? '● EN VIVO' : 'Apagada'; badge.className = on ? 'badge badge-rec' : 'badge badge-live'; }
  if (btn)   btn.textContent = on ? (mode==='cap' ? '⏹ Detener cámara' : '⏹ Detener') : (mode==='cap' ? '📷 Activar cámara' : '▶ Iniciar predicción');
}

/* ═══════════════════════════════════════════
   PREDICCIÓN — resultados WS
═══════════════════════════════════════════ */
function handlePrediction({ letter, confidence, hand_detected, stable }) {
  _updateDetector('cap',  letter, confidence, hand_detected);
  _updateDetector('pred', letter, confidence, hand_detected);

  if (!State.predCamOn) return;
  if (!stable || !letter || letter === '—' || letter === 'Sin modelo') return;

  const now = Date.now();
  const isDifferent = letter !== State.lastStableLetter;
  const cooldownOk  = (now - State.lastStableTime) > State.STABLE_COOLDOWN;

  if (isDifferent || cooldownOk) {
    State.predHistory.push(letter);
    if (State.predHistory.length > 50) State.predHistory.shift();
    _renderPredWord();
    State.lastStableLetter = letter;
    State.lastStableTime   = now;
    State.detectedCount++;
    State.confSum += confidence;
    const sl = document.getElementById('s-letters');
    const sa = document.getElementById('s-acc');
    if (sl) sl.textContent = State.detectedCount;
    if (sa) sa.textContent = Math.round(State.confSum / State.detectedCount) + '%';
    document.querySelectorAll('.alpha-cell').forEach(c => c.classList.remove('detected'));
    const cell = document.getElementById('alpha-' + letter);
    if (cell) { cell.classList.add('detected'); setTimeout(() => cell.classList.remove('detected'), 900); }
  }
}

function _updateDetector(mode, letter, confidence, hand) {
  const s  = mode === 'cap' ? '-cap' : '-pred';
  const ld = document.getElementById('letter-display' + s);
  const cl = document.getElementById('conf-label' + s);
  const cf = document.getElementById('conf-fill' + s);
  if (ld) { ld.textContent = (letter && letter !== '—') ? letter : '—'; ld.className = 'letter-big' + (hand ? '' : ' no-hand'); }
  if (cl) cl.textContent = `Confianza: ${confidence}%`;
  if (cf) cf.style.width = confidence + '%';
}

function _renderPredWord() {
  const el = document.getElementById('pred-word-display');
  if (el) el.innerHTML = State.predHistory.join(' ') + '<span class="cursor-blink"></span>';
}
function clearPredWord()  { State.predHistory = []; State.lastStableLetter = ''; State.lastStableTime = 0; _renderPredWord(); }
function copyPredWord()   { const t = State.predHistory.join(' '); if (!t) { showToast('No hay texto','info'); return; } navigator.clipboard?.writeText(t); showToast('Copiado','success'); }
function speakPredWord()  { const t = State.predHistory.join(' '); if (!t) { showToast('Sin texto','info'); return; } const u = new SpeechSynthesisUtterance(t); u.lang='es-CO'; speechSynthesis.speak(u); }

/* ── Timer ── */
function startTimer() {
  State.seconds = 0; clearInterval(State.timerInterval);
  State.timerInterval = setInterval(() => {
    State.seconds++;
    const el = document.getElementById('s-time');
    if (el) el.textContent = State.seconds + 's';
  }, 1000);
}
function stopTimer() { clearInterval(State.timerInterval); }

/* ═══════════════════════════════════════════
   SEÑAS ENTRENADAS
═══════════════════════════════════════════ */
async function loadTrainedWords() {
  let words = [], collected = [];
  try { const r = await fetch(API_URL+'/api/labels'); const d = await r.json(); words = Array.isArray(d)?d:[]; } catch {}
  try { const r = await fetch(API_URL+'/api/words');  const d = await r.json(); collected = Array.isArray(d)?d:[]; } catch {}
  _renderWordGrids(words, collected);
}

function _renderWordGrids(trained, collected) {
  const wg  = document.getElementById('words-grid');
  const wgc = document.getElementById('words-count');
  if (wg) {
    wg.innerHTML = '';
    if (collected.length === 0) { wg.innerHTML = '<p class="empty-msg">Sin señas capturadas aún.</p>'; }
    else collected.forEach(({ word, samples }) => {
      const c = document.createElement('div'); c.className = 'word-cell';
      c.innerHTML = `<div class="wc-label">${word}</div><div class="wc-samples">${samples} muestras</div>`;
      wg.appendChild(c);
    });
    if (wgc) wgc.textContent = `${collected.length} seña${collected.length!==1?'s':''} capturadas`;
  }
  const pg  = document.getElementById('pred-words-grid');
  const pgc = document.getElementById('pred-words-count');
  if (pg) {
    pg.innerHTML = '';
    if (trained.length === 0) { pg.innerHTML = '<p class="empty-msg">Sin modelo entrenado. Ve a <strong>Entrenar</strong> primero.</p>'; }
    else trained.forEach(w => {
      const c = document.createElement('div'); c.className = 'word-cell';
      c.innerHTML = `<div class="wc-label">${w}</div><div class="wc-samples">Entrenada ✓</div>`;
      pg.appendChild(c);
    });
    if (pgc) pgc.textContent = `${trained.length} seña${trained.length!==1?'s':''} en el modelo`;
  }
  const dg  = document.getElementById('dict-trained-grid');
  const dgc = document.getElementById('dict-trained-count');
  if (dg) {
    dg.innerHTML = '';
    if (trained.length === 0) { dg.innerHTML = '<p class="empty-msg">Sin señas entrenadas aún.</p>'; }
    else trained.forEach(w => {
      const c = document.createElement('div'); c.className = 'alpha-cell'; c.style.padding='13px 7px';
      c.innerHTML = `<div class="alpha-sign" style="font-size:27px">${SIGNS[w]||'🤚'}</div><div class="alpha-letter" style="font-size:13.5px;margin-top:4px">${w}</div>`;
      dg.appendChild(c);
    });
    if (dgc) dgc.textContent = `${trained.length} entrenada${trained.length!==1?'s':''}`;
  }
  const tw = document.getElementById('t-words');
  const ts = document.getElementById('t-samples');
  if (tw) tw.textContent = collected.length;
  if (ts) ts.textContent = collected.length > 0 ? `~${collected.reduce((s,{samples})=>s+samples,0)}` : '—';
}

/* ═══════════════════════════════════════════
   ALFABETO
═══════════════════════════════════════════ */
function renderAlphaGrid(containerId, onClickFn) {
  const grid = document.getElementById(containerId); if (!grid) return;
  grid.innerHTML = '';
  Object.entries(SIGNS).forEach(([l, e]) => {
    const cell = document.createElement('div');
    cell.className = 'alpha-cell'; cell.id = 'alpha-' + l;
    cell.innerHTML = `<div class="alpha-sign">${e}</div><div class="alpha-letter">${l}</div>`;
    if (onClickFn) cell.onclick = () => onClickFn(l);
    grid.appendChild(cell);
  });
}

/* ═══════════════════════════════════════════
   PANEL ENTRENAR
   — muestra solo "Entrenando…" hasta que
     el backend confirma que terminó
═══════════════════════════════════════════ */

/* ═══════════════════════════════════════════
   TRAINING CHART — muestra el PNG generado por matplotlib
═══════════════════════════════════════════ */

function _initTrainChart() {
  _TRAIN_CHARTS.forEach(c => {
    const w = document.getElementById(c.wrap);
    if (w) w.style.display = 'none';
  });
}

const _TRAIN_CHARTS = [
  { wrap: 'train-chart-wrap',  img: 'train-chart-img',  endpoint: '/api/train/chart',              label: 'curvas' },
  { wrap: 'distribution-wrap', img: 'distribution-img', endpoint: '/api/train/class_distribution', label: 'distribución' },
  { wrap: 'metrics-wrap',      img: 'metrics-img',      endpoint: '/api/train/class_metrics',      label: 'métricas por clase' },
  { wrap: 'confusion-wrap',    img: 'confusion-img',    endpoint: '/api/train/confusion',          label: 'top confusiones' },
  { wrap: 'top-errors-wrap',   img: 'top-errors-img',   endpoint: '/api/train/top_errors',         label: 'top errores' },
];

function _showTrainChart() {
  const ts = Date.now();
  _TRAIN_CHARTS.forEach(c => {
    const wrap = document.getElementById(c.wrap);
    const img  = document.getElementById(c.img);
    if (!wrap || !img) return;
    img.src = `${API_URL}${c.endpoint}?t=${ts}`;
    img.onload  = () => { console.log(`[chart] ${c.label} cargada`); wrap.style.display = 'block'; };
    img.onerror = () => { console.warn(`[chart] no se pudo cargar ${c.label}`); wrap.style.display = 'none'; };
  });

  // Mostrar el bloque t-SNE (con selector) y poblar el dropdown con las señas entrenadas
  _populateTsneSelector();
}

async function _populateTsneSelector() {
  const wrap   = document.getElementById('tsne-wrap');
  const select = document.getElementById('tsne-select');
  if (!wrap || !select) return;

  try {
    const r     = await fetch(API_URL + '/api/words');
    const data  = await r.json();
    const words = data.words || [];
    if (words.length === 0) {
      wrap.style.display = 'none';
      return;
    }
    select.innerHTML = '<option value="">— Selecciona una seña —</option>' +
      words.map(w => `<option value="${w}">${w}</option>`).join('');
    wrap.style.display = 'block';
  } catch (err) {
    console.warn('[tsne] no se pudo obtener lista de señas:', err);
    wrap.style.display = 'none';
  }
}

async function _generateTsne() {
  const select = document.getElementById('tsne-select');
  const img    = document.getElementById('tsne-img');
  const status = document.getElementById('tsne-status');
  if (!select || !img) return;

  const word = select.value;
  if (!word) {
    if (status) status.textContent = 'Selecciona una seña primero';
    return;
  }

  if (status) status.textContent = `Generando t-SNE de "${word}"…`;
  img.style.display = 'none';

  const url = `${API_URL}/api/train/tsne?word=${encodeURIComponent(word)}&t=${Date.now()}`;
  img.onload = () => {
    if (status) status.textContent = `✓ t-SNE de "${word}"`;
    img.style.display = 'block';
  };
  img.onerror = () => {
    if (status) status.textContent = `✗ No se pudo generar (¿muy pocas muestras?)`;
    img.style.display = 'none';
  };
  img.src = url;
}

// Conectar botón del t-SNE
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('btn-tsne-generate');
  if (btn) btn.addEventListener('click', _generateTsne);
});

// Compatibilidad con el código viejo que llama _updateTrainChart
function _updateTrainChart() { /* ya no se usa — la imagen se carga al final */ }

function requestTrain() {
  const prog  = document.getElementById('train-prog');
  const lbl   = document.getElementById('train-prog-lbl');
  const badge = document.getElementById('epoch-badge');
  const log   = document.getElementById('train-log');

  // Reset UI
  if (log) log.innerHTML = '';
  _initTrainChart();
  prog.style.width      = '100%';
  prog.style.background = 'var(--teal)';
  prog.style.animation  = 'pulse-bar 1.5s ease-in-out infinite';
  lbl.textContent       = 'Entrenando…';
  badge.textContent     = 'Entrenando…';

  console.log('[train] iniciando…');

  // Leer opciones de los checkboxes
  const force     = document.getElementById('opt-force')?.checked || false;
  const skipTsne  = !(document.getElementById('opt-tsne')?.checked || false);
  console.log('[train] opciones:', { force, skip_tsne: skipTsne });

  fetch(API_URL + '/api/train', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ force, skip_tsne: skipTsne }),
  })
    .then(r => r.json())
    .then(d => {
      console.log('[train] backend respondió:', d);
      if (d.error) {
        _trainError(d.error, prog, lbl, badge);
        return;
      }
      // SIEMPRE hacer polling (más fiable que WS); WS es solo bonus para logs
      _pollTrainDone(prog, lbl, badge);
      _waitForTrainDone(prog, lbl, badge);
    })
    .catch((err) => {
      console.error('[train] error:', err);
      prog.style.animation = '';
      prog.style.width     = '0%';
      lbl.textContent      = 'Usa: python main.py --train';
      badge.textContent    = 'Offline';
      showToast('Backend no disponible', 'error');
    });
}

function _waitForTrainDone(prog, lbl, badge) {
  const ws = new WebSocket(WS_URL.replace('/ws/predict', '/ws/train'));

  ws.onmessage = (e) => {
    const msg = e.data;
    // Update chart on each epoch message
    if (msg.startsWith('Época')) {
      fetch(API_URL + '/api/train/status')
        .then(r => r.json())
        .then(d => { if (d.history) _updateTrainChart(d.history); })
        .catch(() => {});
    }
    // SOLO cerrar al ver "Modelo guardado" (el polling se encarga del resto)
    if (msg.includes('Modelo guardado')) {
      console.log('[train/ws] entrenamiento completado');
      // Pequeño delay para asegurar que el PNG ya está escrito en disco
      setTimeout(_showTrainChart, 600);
      ws.close();
    }
    if (msg.startsWith('✗')) {
      _trainError(msg, prog, lbl, badge);
      ws.close();
    }
  };

  ws.onerror = () => {
    // WS de entrenamiento no disponible — hacer polling simple
    _pollTrainDone(prog, lbl, badge);
  };
}

function _pollTrainDone(prog, lbl, badge) {
  // Fuente primaria de verdad: polling cada 1s a /api/train/status
  let lastEpoch = -1;
  const iv = setInterval(async () => {
    try {
      const r = await fetch(API_URL + '/api/train/status');
      const d = await r.json();

      // Logear solo cuando cambia la época para no spammear consola
      if (d.epoch !== lastEpoch) {
        console.log('[train/status]', { epoch: d.epoch, total: d.total, running: d.running, done: d.done, history_len: d.history?.loss?.length });
        lastEpoch = d.epoch;
      }

      // Actualizar barra de progreso por época
      if (d.epoch && d.total) {
        const pct = (d.epoch / d.total) * 100;
        prog.style.width = pct + '%';
        badge.textContent = `Época ${d.epoch}/${d.total}`;
        lbl.textContent = `Entrenando… ${Math.round(pct)}%`;
      }

      if (d.done) {
        clearInterval(iv);
        prog.style.animation  = '';
        prog.style.width      = '100%';
        prog.style.background = 'linear-gradient(90deg, var(--teal), var(--gold))';
        lbl.textContent       = 'Entrenamiento completado ✓';
        badge.textContent     = 'Completado ✓';
        showToast('✓ Modelo entrenado. Lista actualizada', 'success');
        loadTrainedWords();
        _showTrainChart();
        return;
      }
      if (!d.running && !d.done) {
        clearInterval(iv);
        _trainError('Error en entrenamiento', prog, lbl, badge);
      }
    } catch (_) {}
  }, 3000);
}

function _trainError(msg, prog, lbl, badge) {
  prog.style.animation = '';
  prog.style.width     = '0%';
  prog.style.background = '';
  badge.textContent    = 'Error';
  lbl.textContent      = msg;
  showToast(msg, 'error');
}

function clearLog() {
  const log  = document.getElementById('train-log');
  const prog = document.getElementById('train-prog');
  const lbl  = document.getElementById('train-prog-lbl');
  const badge= document.getElementById('epoch-badge');
  const wrap = document.getElementById('train-chart-wrap');
  if (log)   log.innerHTML    = '';
  if (prog)  { prog.style.width='0%'; prog.style.animation=''; prog.style.background=''; }
  if (lbl)   lbl.textContent  = 'Sin entrenar';
  if (badge) badge.textContent = 'Listo';
  if (wrap)  wrap.style.display = 'none';
  _TRAIN_CHARTS.forEach(c => {
    const w = document.getElementById(c.wrap);
    if (w) w.style.display = 'none';
  });
}

/* ═══════════════════════════════════════════
   DEMO LANDING
═══════════════════════════════════════════ */
const DEMO_WORDS = ['HOLA', 'BIEN', 'MAL', 'POR FAVOR'];
const DEMO_IMGS  = ['/static/hola.png', '/static/bien.png', '/static/mal.png', '/static/porfavor.png'];

function startDemoLoop() {
  setInterval(() => {
    State.demoIdx = (State.demoIdx + 1) % DEMO_WORDS.length;
    const w   = DEMO_WORDS[State.demoIdx];
    const img = DEMO_IMGS[State.demoIdx];
    const phWord = document.getElementById('ph-word');
    if (phWord) { phWord.style.opacity='0'; setTimeout(()=>{phWord.textContent=w;phWord.style.opacity='1';phWord.style.transition='opacity .3s';},200); }
    const imgEl = document.getElementById('ph-hand-img');
    if (imgEl) { imgEl.style.opacity='0'; setTimeout(()=>{imgEl.src=img;imgEl.style.opacity='1';},400); }
    const chip = document.getElementById('ph-chip');
    if (chip) { chip.style.opacity='0'; setTimeout(()=>{chip.textContent=w;chip.style.opacity='1';chip.style.transition='opacity .3s';},200); }
  }, 2800);
}

function speakDemo() {
  const w = document.getElementById('demo-word')?.textContent;
  if (w && 'speechSynthesis' in window) { const u=new SpeechSynthesisUtterance(w.replace(/_/g,' ')); u.lang='es-CO'; speechSynthesis.speak(u); }
}

/* ═══════════════════════════════════════════
   UTILIDADES
═══════════════════════════════════════════ */
function showToast(msg, type='info') {
  const t = document.getElementById('toast');
  t.textContent=msg; t.className='toast '+type; t.style.display='block';
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(()=>t.style.display='none', 3000);
}

function initReveal() {
  const obs = new IntersectionObserver(entries => {
    entries.forEach((e,i)=>{ if(e.isIntersecting){setTimeout(()=>e.target.classList.add('visible'),i*80);obs.unobserve(e.target);} });
  }, {threshold:0.1});
  document.querySelectorAll('.reveal').forEach(el=>obs.observe(el));
}

/* ═══════════════════════════════════════════
   INIT
═══════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  renderAlphaGrid('alpha-grid-cap', null);
  renderAlphaGrid('dict-alpha-grid', null);
  setupCapture();
  loadTrainedWords();
  startDemoLoop();
  initReveal();
  document.getElementById('btn-cam-pred')?.addEventListener('click', togglePredCamera);
  document.getElementById('btn-clear-pred')?.addEventListener('click', clearPredWord);
  document.getElementById('btn-copy-pred')?.addEventListener('click', copyPredWord);
  document.getElementById('btn-speak-pred')?.addEventListener('click', speakPredWord);
  document.getElementById('btn-train')?.addEventListener('click', requestTrain);
  document.getElementById('btn-clear-log')?.addEventListener('click', clearLog);
});