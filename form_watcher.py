"""Invisible component that reports the review form's pending values and previews results.

Streamlit form widgets send nothing until the form is submitted, so the server
can't see what the user is typing or which model they just picked. This
component (``st.components.v2``, built into Streamlit; no extra dependency)
mounts in the page DOM, finds the form's own widgets by their accessible names,
and reports them with ``setStateValue("pending", {text, model})``:

- text: 1 s after the user stops typing (debounced ``input`` events)
- model: immediately when the "Model" combobox value changes

Each report reruns the script, which starts a background prefetch. When the
prefetch is still running, the script answers ``waiting`` and the component
asks again with a short ``poll`` trigger. A finished outcome comes back as
``data.ready``. When Analyze is clicked (or Ctrl/Cmd+Enter is pressed in the
review) and the form still holds exactly that text and model, the component
shows the ready result at once while the normal submit rerun renders the real
Streamlit elements, then hides it again. Nothing is shown before the click.

The ready result is plain text set with ``textContent``, never HTML.
"""

import streamlit as st

KEY = "form_watch"

_HTML = """
<div class="rp-instant" data-testid="instantResult" hidden aria-live="polite">
  <div class="rp-label"></div>
  <div class="rp-metric"><span class="rp-metric-name">Predicted-class probability</span>
    <span class="rp-metric-value"></span></div>
  <div class="rp-bars"></div>
  <div class="rp-breakdown"></div>
</div>
"""

_CSS = """
.rp-instant { font-family: inherit; color: inherit; }
.rp-instant[hidden] { display: none; }
.rp-label { font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0; }
.rp-metric-name { display: block; font-size: 0.875rem; opacity: 0.7; }
.rp-metric-value { display: block; font-size: 2.25rem; line-height: 1.2; }
.rp-bars { margin: 0.75rem 0; display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 0.75rem; }
.rp-bar-name { font-size: 0.8rem; opacity: 0.8; }
.rp-bar-track { align-self: center; height: 0.9rem; }
.rp-bar-fill { height: 100%; background: #bb0000; border-radius: 2px; }
.rp-breakdown { font-size: 0.875rem; opacity: 0.7; }
"""

_JS = """
const TEXT = 'textarea[aria-label="Review text"]';
const MODEL = 'input[role="combobox"][aria-label="Model"]';
const SUBMIT = '[data-testid="stFormSubmitButton"] button';
const DEBOUNCE_MS = 1000;
const MODEL_CHECK_MS = 250;
const POLL_MS = 300;
const MAX_POLLS = 40;

function current() {
  const text = document.querySelector(TEXT);
  const model = document.querySelector(MODEL);
  return { text: text ? text.value : '', model: model ? model.value : null };
}

function same(a, b) {
  return !!a && !!b && a.text === b.text && a.model === b.model;
}

function wire(w) {
  w.report = () => {
    clearTimeout(w.debounce);
    w.polls = 0;
    w.setState('pending', current());
  };
  document.addEventListener('input', (e) => {
    if (e.target.matches && e.target.matches(TEXT)) {
      w.hide();
      clearTimeout(w.debounce);
      w.debounce = setTimeout(w.report, DEBOUNCE_MS);
    }
  }, true);
  w.model = current().model;
  setInterval(() => {
    const box = document.querySelector(MODEL);
    // While the list is open the input holds the type-to-filter text, not a choice.
    if (box && box.getAttribute('aria-expanded') === 'true') return;
    const model = current().model;
    if (model !== w.model) {
      w.model = model;
      w.hide();
      w.report();
    }
  }, MODEL_CHECK_MS);
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest(SUBMIT)) w.reveal();
  }, true);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && e.target.matches && e.target.matches(TEXT)) w.reveal();
  }, true);
}

function show(root, view, labels) {
  root.querySelector('.rp-label').textContent = view.label;
  const hasProbs = Array.isArray(view.probs);
  root.querySelector('.rp-metric').hidden = !hasProbs;
  root.querySelector('.rp-breakdown').textContent = view.breakdown || '';
  const bars = root.querySelector('.rp-bars');
  bars.replaceChildren();
  if (hasProbs) {
    root.querySelector('.rp-metric-value').textContent = view.metric;
    view.probs.forEach((p, i) => {
      const name = document.createElement('span');
      name.className = 'rp-bar-name';
      name.textContent = labels[i] || '';
      const track = document.createElement('div');
      track.className = 'rp-bar-track';
      const fill = document.createElement('div');
      fill.className = 'rp-bar-fill';
      fill.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
      track.append(fill);
      bars.append(name, track);
    });
  }
}

export default function (component) {
  const { setStateValue, setTriggerValue, data, parentElement } = component;
  const w = (window.__reviewPrefetch ??= { polls: 0 });
  w.setState = setStateValue;
  w.setTrigger = setTriggerValue;
  w.data = data;
  w.root = parentElement;
  w.box = () => w.root && w.root.querySelector('.rp-instant');
  w.hide = () => { const box = w.box(); if (box) box.hidden = true; };
  w.reveal = () => {
    const ready = w.data && w.data.ready;
    const box = w.box();
    if (!box || !ready || !same(ready, current())) return;
    show(w.root, ready.view, w.data.labels || []);
    box.hidden = false;
    w.revealedAt = performance.now();
  };
  if (!w.wired) {
    w.wired = true;
    wire(w);
  }
  // The server rendered its own result this run (or the form moved on): drop the preview.
  if (data.submitted || !same(data.ready, current())) w.hide();

  clearTimeout(w.pollTimer);
  if (data.waiting && w.polls < MAX_POLLS) {
    w.pollTimer = setTimeout(() => { w.polls += 1; w.setTrigger('poll', w.polls); }, POLL_MS);
  }
}
"""

_component = st.components.v2.component(
    "review_form_watcher", html=_HTML, css=_CSS, js=_JS, isolate_styles=False
)


def pending() -> dict | None:
    """The last ``{text, model}`` the browser reported, if well-formed."""
    state = st.session_state.get(KEY) or {}
    value = state.get("pending") if hasattr(state, "get") else None
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        return None
    model = value.get("model")
    return {"text": value["text"], "model": model if isinstance(model, str) else None}


def mount(*, data: dict, on_report) -> None:
    """Mount the watcher. ``on_report`` runs before reruns the watcher causes."""
    _component(
        key=KEY,
        data=data,
        default={"pending": None},
        on_pending_change=on_report,
        on_poll_change=on_report,
    )
