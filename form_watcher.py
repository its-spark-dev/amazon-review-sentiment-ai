"""Invisible component that reports the review form's pending values and flips in results.

Streamlit form widgets send nothing until the form is submitted, so the server
can't see what the user is typing or which model they just picked. This
component (``st.components.v2``, built into Streamlit; no extra dependency)
mounts in the page DOM, finds the form's own widgets by their accessible names,
and reports them with ``setStateValue("pending", {text, model})``:

- text: 1 s after the user stops typing (debounced ``input`` events)
- model: immediately when the "Model" combobox value changes

Each report reruns the script, which starts a background prefetch. When the
prefetch is still running, the script answers ``waiting`` and the component
asks again with a short ``poll`` trigger. A finished outcome is rendered by the
script into the hidden back result slot (double buffering: the browser lays out
and draws it, chart included, without showing it) and announced as
``data.ready``. When Analyze is clicked (or Ctrl/Cmd+Enter is pressed in the
review) and the form still holds exactly that text and model, the component
flips the back slot into view at once. The submit run then renders the same
elements into that same slot, and the override is removed. Nothing is shown
before the click.
"""

import streamlit as st

KEY = "form_watch"

_HTML = "<div hidden></div>"

_JS = """
const TEXT = 'textarea[aria-label="Review text"]';
const MODEL = 'input[role="combobox"][aria-label="Model"]';
const SUBMIT = '[data-testid="stFormSubmitButton"] button';
const DEBOUNCE_MS = 1000;
const MODEL_CHECK_MS = 250;
const POLL_MS = 300;
const MAX_POLLS = 40;
const QUIET_MAX_MS = 5000;

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
    if (w.submitting) return;
    w.polls = 0;
    w.setState('pending', current());
  };
  document.addEventListener('input', (e) => {
    if (e.target.matches && e.target.matches(TEXT)) {
      clearTimeout(w.debounce);
      w.debounce = setTimeout(w.report, DEBOUNCE_MS);
    }
  }, true);
  w.model = current().model;
  // Text typed before this script loaded raised no event we saw: treat it as fresh typing.
  if (current().text.trim()) w.debounce = setTimeout(w.report, DEBOUNCE_MS);
  setInterval(() => {
    const box = document.querySelector(MODEL);
    // While the list is open the input holds the type-to-filter text, not a choice.
    if (box && box.getAttribute('aria-expanded') === 'true') return;
    const model = current().model;
    if (model !== w.model) {
      w.model = model;
      w.report();
    }
  }, MODEL_CHECK_MS);
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest(SUBMIT)) w.onSubmit();
  }, true);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && e.target.matches && e.target.matches(TEXT)) w.onSubmit();
  }, true);
}

// Flip the buffered back slot visible (and the front slot away) until the server's
// submit render, which puts the same elements in that slot without hiding it.
function flip(w, on) {
  let style = document.getElementById('rp-flip');
  if (!on) { if (style) style.remove(); return; }
  if (!style) {
    style = document.createElement('style');
    style.id = 'rp-flip';
    document.head.append(style);
  }
  const back = w.data.back, front = w.data.front;
  style.textContent =
    `html body .st-key-${back} { visibility: visible !important; height: auto !important; max-height: none !important; ` +
    `overflow: visible !important; pointer-events: auto !important; }` +
    `html body [data-testid=stLayoutWrapper]:has(> .st-key-${back}) { margin-bottom: 0 !important; }` +
    `html body .st-key-${front} { display: none !important; }`;
}

export default function (component) {
  const { setStateValue, setTriggerValue, data, parentElement } = component;
  const w = (window.__reviewPrefetch ??= { polls: 0 });
  w.setState = setStateValue;
  w.setTrigger = setTriggerValue;
  w.data = data;
  w.reveal = () => {
    const ready = w.data && w.data.ready;
    if (!ready || !same(ready, current()) || !/^[a-z_]+$/.test(w.data.back + w.data.front)) return;
    flip(w, true);
  };
  // A rerun requested while the submit's run is executing would interrupt it and drop the
  // submit, so stay quiet from the click until the server's submit render arrives.
  w.onSubmit = () => {
    w.submitting = true;
    clearTimeout(w.debounce);
    clearTimeout(w.pollTimer);
    clearTimeout(w.quietTimer);
    w.quietTimer = setTimeout(() => { w.submitting = false; }, QUIET_MAX_MS);
    w.model = current().model;  // the submit carries this choice; don't report it again
    w.reveal();
  };
  if (!w.wired) {
    w.wired = true;
    wire(w);
  }
  // The server rendered the result into the flipped slot this run: the override can go.
  if (data.submitted) {
    w.submitting = false;
    clearTimeout(w.quietTimer);
    requestAnimationFrame(() => flip(w, false));
  }

  clearTimeout(w.pollTimer);
  if (data.waiting && w.polls < MAX_POLLS && !w.submitting) {
    w.pollTimer = setTimeout(() => {
      if (w.submitting) return;
      w.polls += 1;
      w.setTrigger('poll', w.polls);
    }, POLL_MS);
  }
}
"""

_component = st.components.v2.component(
    "review_form_watcher", html=_HTML, js=_JS, isolate_styles=False
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
