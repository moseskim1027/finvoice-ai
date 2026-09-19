const form = document.querySelector('#conversation-form');
const input = document.querySelector('#message-input');
const slider = document.querySelector('#confidence');
const sliderValue = document.querySelector('#confidence-value');
const messages = document.querySelector('#messages');
const trace = document.querySelector('#trace');
const emptyState = document.querySelector('#empty-state');
const sessionId = `local-demo-${crypto.randomUUID()}`;

slider.addEventListener('input', () => { sliderValue.textContent = Number(slider.value).toFixed(2); });

function addMessage(kind, text, detail = '') {
  const item = document.createElement('div');
  item.className = `message ${kind}`;
  item.innerHTML = kind === 'agent'
    ? `<span class="avatar">F</span><div><p></p><small></small></div>`
    : '<div><p></p></div>';
  item.querySelector('p').textContent = text;
  if (detail) item.querySelector('small').textContent = detail;
  messages.append(item);
  messages.scrollTop = messages.scrollHeight;
}

function renderTrace(data) {
  emptyState.classList.add('hidden');
  trace.classList.remove('hidden');
  const isEscalation = data.decision === 'escalate';
  document.querySelector('#decision').textContent = isEscalation ? 'ESCALATE' : 'RESPOND';
  document.querySelector('#decision').className = isEscalation ? 'danger' : 'safe';
  document.querySelector('#reason').textContent = data.reason ? data.reason.replaceAll('_', ' ') : 'approved context';
  document.querySelector('#model').textContent = data.provider?.model ?? 'policy handoff';
  document.querySelector('#sources').textContent = data.provider?.retrieved_documents ?? 0;
  const confidence = data.confidence ?? 0;
  document.querySelector('#response-confidence').textContent = `${Math.round(confidence * 100)}%`;
  document.querySelector('#confidence-bar').style.width = `${confidence * 100}%`;
  document.querySelector('#request-id').textContent = `Trace ID ${data.request_id}`;
  const citations = document.querySelector('#citations');
  citations.replaceChildren();
  data.citations.forEach((citation) => {
    const item = document.createElement('article');
    item.innerHTML = `<span>APPROVED SOURCE</span><strong></strong><p></p>`;
    item.querySelector('strong').textContent = citation.title;
    item.querySelector('p').textContent = citation.excerpt;
    citations.append(item);
  });
}

async function submit(message, confidence) {
  addMessage('user', message);
  input.value = '';
  const button = form.querySelector('button');
  button.disabled = true;
  button.textContent = 'Thinking…';
  try {
    const response = await fetch('/v1/conversations/respond', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message, confidence }),
    });
    if (!response.ok) throw new Error('The local service could not complete the simulation.');
    const data = await response.json();
    addMessage('agent', data.message, data.decision === 'escalate' ? 'Safety handoff selected' : 'Grounded local response');
    renderTrace(data);
  } catch (error) {
    addMessage('agent', error.message, 'Local simulation error');
  } finally {
    button.disabled = false;
    button.innerHTML = 'Send <span>↗</span>';
    input.focus();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (message) submit(message, Number(slider.value));
});
document.querySelectorAll('.scenario').forEach((button) => button.addEventListener('click', () => {
  slider.value = button.dataset.confidence;
  slider.dispatchEvent(new Event('input'));
  submit(button.dataset.message, Number(button.dataset.confidence));
}));
document.querySelector('#reset').addEventListener('click', () => {
  messages.querySelectorAll('.message:not(:first-child)').forEach((message) => message.remove());
  trace.classList.add('hidden'); emptyState.classList.remove('hidden'); input.focus();
});

function pcmWav(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2); const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  write(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); write(8, 'WAVE'); write(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, sample, true)); return buffer;
}
function sampleAudio(seconds = .72) { return Array.from({ length: Math.round(16000 * seconds) }, (_, index) => Math.round(1700 * Math.sin(index / 13))); }
function show(id, data) { document.querySelector(id).textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2); }
document.querySelector('#analyze-speech').addEventListener('click', async () => {
  show('#speech-result', 'Analyzing bounded PCM WAV…');
  const formData = new FormData(); formData.append('file', new Blob([pcmWav(sampleAudio())], { type: 'audio/wav' }), 'synthetic.wav');
  const response = await fetch('/v1/audio/analyze', { method: 'POST', body: formData }); show('#speech-result', await response.json());
});
document.querySelector('#run-stream').addEventListener('click', () => {
  const result = '#stream-result'; show(result, 'Connecting…'); const id = `stream-demo-${crypto.randomUUID()}`;
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/v1/audio/stream/${id}`); const events = [];
  socket.onmessage = ({ data }) => { const event = JSON.parse(data); if (event.type !== 'accepted') events.push(event); if (event.type === 'finalized') { show(result, events); socket.close(); } };
  socket.onopen = () => { [...Array(3).fill(1100), ...Array(15).fill(0)].forEach((value, sequence) => { const pcm = new Int16Array(320).fill(value); const bytes = new Uint8Array(pcm.buffer); socket.send(JSON.stringify({ type: 'audio_chunk', utterance_id: 'guided-demo', sequence, sample_rate_hz: 16000, pcm_s16le_base64: btoa(String.fromCharCode(...bytes)) })); }); };
  socket.onerror = () => show(result, 'WebSocket connection failed. Is the local API running?');
});
async function tool(name, arguments, confirmed = false) { const response = await fetch(`/v1/demo/tools/${name}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ arguments, confirmed }) }); show('#tool-result', await response.json()); }
document.querySelector('#read-tool').addEventListener('click', () => tool('get_demo_account_status', { account_id: 'DEMO-001' }));
document.querySelector('#write-tool').addEventListener('click', () => tool('create_demo_support_ticket', { session_id: 'browser-demo', category: 'technical' }, true));
