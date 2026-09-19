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
