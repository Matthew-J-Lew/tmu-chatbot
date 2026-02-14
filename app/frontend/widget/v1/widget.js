/*
  TMU Arts Chatbot Widget (v1)

  - Dependency-free Web Component
  - Shadow DOM: no global CSS
  - Embeddable via <script src=".../widget/v1/widget.js"></script>

  Public endpoint: POST  {apiBaseUrl}/api/chat
  Admin endpoint:  POST  {apiBaseUrl}/admin/tools/chat  (includes debug payload)

  Example embed:

  <div id="tmu-chat"></div>
  <script src="https://YOUR_HOST/widget/v1/widget.js" defer></script>
  <script>
    window.TMUChatbot.init({
      container: '#tmu-chat',
      apiBaseUrl: 'https://YOUR_HOST',
      mode: 'public',
      title: 'TMU Arts Chat',
      enableCitations: true
    });
  </script>
*/

(function () {
  'use strict';

  const DEFAULTS = {
    apiBaseUrl: '',
    mode: 'public', // 'public' | 'admin'
    title: 'TMU Arts Chat',
    initialPrompt: '',
    enableCitations: true,
    enableDebug: false,
    defaultParams: {},
    // display: 'floating' shows a chat bubble in the corner (default).
    // display: 'inline' renders the full panel where it is mounted.
    display: 'floating',
    // Optional override for launcher icon image (defaults to tmu_logo.png next to widget.js)
    launcherIconUrl: "",
    // If container is omitted, the widget will be appended to <body>.
    container: null,
  };

  function normalizeBaseUrl(url) {
    if (!url) return '';
    return String(url).replace(/\/+$/, '');
  }

  function detectAssetBase() {
    try {
      const cs = document.currentScript;
      if (cs && cs.src) return cs.src.replace(/\/[^\/]+$/, "/");
    } catch (_e) {}
    try {
      const scripts = Array.from(document.getElementsByTagName("script"));
      const hit = scripts.find(sc => sc && sc.src && /\/widget\.js(\?|#|$)/.test(sc.src));
      if (hit && hit.src) return hit.src.replace(/\/[^\/]+$/, "/");
    } catch (_e) {}
    return "";
  }

  const ASSET_BASE = detectAssetBase();
  const DEFAULT_LAUNCHER_ICON_URL = (ASSET_BASE ? (ASSET_BASE + "tmu_logo.png") : "tmu_logo.png");


  function parseBool(val, fallback) {
    if (val === undefined || val === null || val === '') return fallback;
    const s = String(val).trim().toLowerCase();
    return s === '1' || s === 'true' || s === 'yes' || s === 'on';
  }

  function uuid() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    // Fallback (not cryptographically strong)
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function safeJson(obj) {
    try {
      return JSON.stringify(obj, null, 2);
    } catch (_e) {
      return String(obj);
    }
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (e) {
      data = { raw: text };
    }

    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) ? (data.detail || data.message) : text;
      throw new Error(detail || `HTTP ${res.status}`);
    }

    return data;
  }

  async function sendMessage(apiBaseUrl, mode, question, sessionId, options) {
    const base = normalizeBaseUrl(apiBaseUrl);
    if (!base) throw new Error('Missing apiBaseUrl');

    const endpoint = mode === 'admin' ? '/admin/tools/chat' : '/api/chat';
    const url = base + endpoint;

    // We keep the server contract simple: it only needs `question`.
    // For admin calls, we allow optional params to tune retrieval.
    const body = mode === 'admin'
      ? { question, params: (options && options.params) ? options.params : undefined }
      : { question };

    // sessionId is currently client-only, but useful for future analytics.
    // (If you later add server-side session tracking, you can include it in headers.)
    return await postJson(url, body);
  }

  const STYLE = `
    :host { display: block; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

    /* Root container */
    .root {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 2147483647;
    }
    .root.inline { position: relative; right: auto; bottom: auto; }

    /* Launcher bubble */
    .launcher {
      width: 52px;
      height: 52px;
      border-radius: 999px;
      border: 2px solid #000;
      background: #1d4a97;
      color: white;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 10px 24px rgba(0,0,0,0.22);
      cursor: pointer;
      padding: 0;
      user-select: none;
    }
    .launcher:hover { filter: brightness(0.98); }
    .launcher:active { transform: translateY(1px); }
    .launcherImg { width: 30px; height: 30px; object-fit: contain; display: block; }

    /* Chat panel */
    .panel {
      position: absolute;
      right: 0;
      bottom: 64px;
      width: min(380px, calc(100vw - 40px));
      height: 560px;
      max-height: min(70vh, 560px);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
      box-shadow: 0 18px 40px rgba(0,0,0,0.22);
      display: flex;
      flex-direction: column;
    }
    .panel.hidden { display: none; }

    /* Inline mode: render panel in flow, no launcher */
    .root.inline .panel { position: static; width: 100%; height: 560px; max-height: 80vh; box-shadow: none; }
    .root.inline .launcher { display: none; }

    .header { padding: 12px 14px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(0,0,0,0.08); background: #1d4a97; color: #fff; }
    .title { font-size: 14px; font-weight: 650; }
    .actions { display: flex; gap: 8px; }
    button { font: inherit; cursor: pointer; border-radius: 10px; border: 1px solid rgba(0,0,0,0.15); background: #fff; color: rgba(0,0,0,0.9); padding: 6px 10px; font-size: 12px; }
    button:hover { background: rgba(0,0,0,0.03); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .body { flex: 1; overflow: auto; padding: 12px; background: #fff; }
    .msg { display: flex; flex-direction: column; gap: 6px; margin: 10px 0; }
    .bubble { max-width: 95%; padding: 10px 12px; border-radius: 12px; font-size: 13px; line-height: 1.35; white-space: pre-wrap; word-wrap: break-word; }
    .user { align-items: flex-end; }
    .user .bubble { background: rgba(13,110,253,0.12); border: 1px solid rgba(13,110,253,0.18); }
    .assistant { align-items: flex-start; }
    .assistant .bubble { background: rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.08); }
    .meta { display: flex; align-items: center; gap: 10px; font-size: 11px; color: rgba(0,0,0,0.65); }
    .citations { margin: 0; padding-left: 18px; font-size: 12px; }
    .citations li { margin: 2px 0; }
    .citations a { color: #0d6efd; text-decoration: none; }
    .citations a:hover { text-decoration: underline; }
    /* Footer (input area). We tint the whole footer while keeping controls unchanged. */
    .footer { padding: 12px; border-top: 1px solid rgba(255,255,255,0.22); background: #1d4a97; }
    .row { display: flex; gap: 8px; }
    input[type="text"] { flex: 1; font: inherit; border-radius: 10px; border: 1px solid rgba(0,0,0,0.15); padding: 10px 12px; font-size: 13px; }
    .hint { margin-top: 10px; font-size: 11px; color: #fff; background: #1d4a97; padding: 6px 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.22); }
    .spinner { display: inline-block; width: 14px; height: 14px; border-radius: 999px; border: 2px solid rgba(0,0,0,0.25); border-top-color: rgba(0,0,0,0.65); animation: spin 1s linear infinite; }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .debugWrap { margin-top: 6px; width: 100%; }
    .debugToggle { font-size: 11px; padding: 4px 8px; border-radius: 999px; }
    pre { margin: 8px 0 0 0; padding: 10px 12px; border-radius: 10px; background: rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.08); overflow: auto; font-size: 11px; }
  `;

  class TMUChatWidget extends HTMLElement {
    static get observedAttributes() {
      return [
        'api-base-url',
        'mode',
        'title',
        'initial-prompt',
        'enable-citations',
        'enable-debug',
        'display',
      ];
    }

    constructor() {
      super();
      this._cfg = { ...DEFAULTS };
      this._shadow = this.attachShadow({ mode: 'open' });
      this._messages = [];
      this._sending = false;
      this._sessionId = null;

      // UI state
      this._open = false;

      this._els = {};
    }

    connectedCallback() {
      this._sessionId = this._getOrCreateSessionId();
      this._readAttrsIntoConfig();
      this._render();

      if (this._cfg.initialPrompt) {
        this._appendAssistant(this._cfg.initialPrompt, { sources: null, debug: null, meta: null });
      }
    }

    attributeChangedCallback() {
      // When attributes change, re-read config and re-render header/footer.
      // We intentionally do NOT wipe message state.
      this._readAttrsIntoConfig();
      this._updateHeader();
    }

    setConfig(cfg) {
      this._cfg = { ...this._cfg, ...cfg };
      this._cfg.apiBaseUrl = normalizeBaseUrl(this._cfg.apiBaseUrl);
      this._updateHeader();
      this._updatePanelVisibility();
    }

    _getOrCreateSessionId() {
      try {
        const key = 'tmu_chatbot_session_id';
        const existing = window.localStorage.getItem(key);
        if (existing) return existing;
        const id = uuid();
        window.localStorage.setItem(key, id);
        return id;
      } catch (_e) {
        return uuid();
      }
    }

    _readAttrsIntoConfig() {
      const a = (name) => this.getAttribute(name);

      const apiBaseUrl = a('api-base-url');
      const mode = a('mode');
      const title = a('title');
      const initialPrompt = a('initial-prompt');
      const enableCitations = a('enable-citations');
      const enableDebug = a('enable-debug');
      const display = a('display');

      if (apiBaseUrl) this._cfg.apiBaseUrl = normalizeBaseUrl(apiBaseUrl);
      if (mode) this._cfg.mode = mode === 'admin' ? 'admin' : 'public';
      if (title) this._cfg.title = title;
      if (initialPrompt) this._cfg.initialPrompt = initialPrompt;
      this._cfg.enableCitations = parseBool(enableCitations, this._cfg.enableCitations);
      this._cfg.enableDebug = parseBool(enableDebug, this._cfg.enableDebug);
      if (display) this._cfg.display = display === 'inline' ? 'inline' : 'floating';
    }

    _render() {
      this._shadow.innerHTML = '';

      const style = document.createElement('style');
      style.textContent = STYLE;

      const root = document.createElement('div');
      root.className = `root ${this._cfg.display === 'inline' ? 'inline' : ''}`.trim();

      const launcher = document.createElement('button');
      launcher.className = 'launcher';
      launcher.setAttribute('aria-label', 'Open chat');
      // icon set in _setLauncherIcon()
      launcher.addEventListener('click', () => this.toggle());

      const panel = document.createElement('div');
      panel.className = `panel ${this._cfg.display === 'inline' ? '' : 'hidden'}`.trim();

      const header = document.createElement('div');
      header.className = 'header';

      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = this._cfg.title;

      const actions = document.createElement('div');
      actions.className = 'actions';
      const btnClose = document.createElement('button');
      btnClose.textContent = 'Close';
      btnClose.addEventListener('click', () => this.close());

      const btnReset = document.createElement('button');
      btnReset.textContent = 'Reset';
      btnReset.addEventListener('click', () => this.reset());

      const btnCopy = document.createElement('button');
      btnCopy.textContent = 'Copy answer';
      btnCopy.addEventListener('click', () => this.copyLastAnswer());

      actions.appendChild(btnCopy);
      actions.appendChild(btnReset);
      actions.appendChild(btnClose);

      header.appendChild(title);
      header.appendChild(actions);

      const body = document.createElement('div');
      body.className = 'body';

      const footer = document.createElement('div');
      footer.className = 'footer';

      const row = document.createElement('div');
      row.className = 'row';

      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Ask a question…';
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.send();
        }
      });

      const btnSend = document.createElement('button');
      btnSend.textContent = 'Send';
      btnSend.addEventListener('click', () => this.send());

      row.appendChild(input);
      row.appendChild(btnSend);

      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = this._cfg.mode === 'admin' ? 'Admin mode: debug details may be available.' : 'Answers are based on official TMU Arts sources.';

      footer.appendChild(row);
      footer.appendChild(hint);

      panel.appendChild(header);
      panel.appendChild(body);
      panel.appendChild(footer);

      root.appendChild(panel);
      root.appendChild(launcher);

      this._shadow.appendChild(style);
      this._shadow.appendChild(root);

      this._els = { root, launcher, panel, header, title, actions, btnReset, btnCopy, btnClose, body, footer, input, btnSend, hint };

      this._launcherIconUrl = this._cfg.launcherIconUrl || DEFAULT_LAUNCHER_ICON_URL;
      this._setLauncherIcon(this._open);

      this._renderMessages();
      this._updatePanelVisibility();
    }


    _setLauncherIcon(isOpen) {
      if (!this._els.launcher) return;
      const btn = this._els.launcher;
      while (btn.firstChild) btn.removeChild(btn.firstChild);

      if (isOpen) {
        const span = document.createElement("span");
        span.textContent = "✕";
        span.style.fontSize = "18px";
        span.style.lineHeight = "1";
        btn.appendChild(span);
      } else {
        const img = document.createElement("img");
        img.className = "launcherImg";
        img.alt = "TMU";
        img.src = this._launcherIconUrl || DEFAULT_LAUNCHER_ICON_URL;
        btn.appendChild(img);
      }
    }

    _updateHeader() {
      if (!this._els.title) return;
      this._els.title.textContent = this._cfg.title;
      if (this._els.hint) {
        this._els.hint.textContent = this._cfg.mode === 'admin'
          ? 'Admin mode: debug details may be available.'
          : 'Answers are based on official TMU Arts sources.';
      }
    }

    _updatePanelVisibility() {
      if (!this._els.panel || !this._els.launcher || !this._els.root) return;
      const inline = this._cfg.display === "inline";
      this._els.root.className = `root ${inline ? "inline" : ""}`.trim();

      if (inline) {
        this._els.panel.classList.remove("hidden");
        this._els.launcher.setAttribute("aria-hidden", "true");
        return;
      }

      // floating
      if (this._open) this._els.panel.classList.remove("hidden");
      else this._els.panel.classList.add("hidden");
      this._els.launcher.setAttribute("aria-label", this._open ? "Close chat" : "Open chat");
      this._setLauncherIcon(this._open);
    }

    open() {
      if (this._cfg.display === 'inline') return;
      this._open = true;
      this._updatePanelVisibility();
      // focus input for convenience
      setTimeout(() => { try { this._els.input && this._els.input.focus(); } catch (_e) {} }, 0);
    }

    close() {
      if (this._cfg.display === 'inline') return;
      this._open = false;
      this._updatePanelVisibility();
    }

    toggle() {
      if (this._cfg.display === 'inline') return;
      this._open ? this.close() : this.open();
    }

    reset() {
      this._messages = [];
      this._sending = false;
      this._renderMessages();

      if (this._cfg.initialPrompt) {
        this._appendAssistant(this._cfg.initialPrompt, { sources: null, debug: null, meta: null });
      }
    }

    async copyLastAnswer() {
      const last = [...this._messages].reverse().find(m => m.role === 'assistant' && m.text);
      if (!last) return;

      try {
        await navigator.clipboard.writeText(last.text);
      } catch (_e) {
        // Fallback: create temp textarea
        const ta = document.createElement('textarea');
        ta.value = last.text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
    }

    async send() {
      if (!this._els.input) return;
      const question = String(this._els.input.value || '').trim();
      if (!question || this._sending) return;

      this._els.input.value = '';
      this._appendUser(question);

      // Placeholder assistant message with spinner
      const placeholderId = uuid();
      this._appendAssistant('', { pending: true, id: placeholderId, sources: null, debug: null, meta: null });

      this._sending = true;
      this._setSendDisabled(true);

      try {
        const params = (this._cfg.defaultParams && typeof this._cfg.defaultParams === 'object')
          ? this._cfg.defaultParams
          : {};

        const resp = await sendMessage(
          this._cfg.apiBaseUrl,
          this._cfg.mode,
          question,
          this._sessionId,
          { params }
        );

        const answer = (resp && resp.answer) ? String(resp.answer) : '';
        const sources = (resp && resp.sources) ? resp.sources : null;
        const debug = (resp && resp.debug) ? resp.debug : null;
        const meta = {
          latency_ms: resp && resp.latency_ms,
          cached: resp && resp.cached,
          timings: resp && resp.timings,
        };

        this._replaceAssistant(placeholderId, answer, { sources, debug, meta });
      } catch (err) {
        const msg = (err && err.message) ? err.message : 'Request failed.';
        this._replaceAssistant(placeholderId, `Sorry — I couldn't complete that request.\n\n${msg}`, { sources: null, debug: null, meta: null });
      } finally {
        this._sending = false;
        this._setSendDisabled(false);
      }
    }

    _setSendDisabled(disabled) {
      if (this._els.btnSend) this._els.btnSend.disabled = disabled;
      if (this._els.input) this._els.input.disabled = disabled;
      if (this._els.btnCopy) this._els.btnCopy.disabled = disabled && this._messages.length === 0;
    }

    _appendUser(text) {
      this._messages.push({ role: 'user', text: String(text) });
      this._renderMessages();
    }

    _appendAssistant(text, extra) {
      const m = {
        role: 'assistant',
        text: String(text || ''),
        pending: !!(extra && extra.pending),
        id: (extra && extra.id) ? extra.id : uuid(),
        sources: (extra && 'sources' in extra) ? extra.sources : null,
        debug: (extra && 'debug' in extra) ? extra.debug : null,
        meta: (extra && 'meta' in extra) ? extra.meta : null,
      };
      this._messages.push(m);
      this._renderMessages();
      return m.id;
    }

    _replaceAssistant(id, text, extra) {
      const idx = this._messages.findIndex(m => m.role === 'assistant' && m.id === id);
      if (idx === -1) return;
      const prev = this._messages[idx];
      this._messages[idx] = {
        ...prev,
        text: String(text || ''),
        pending: false,
        sources: (extra && 'sources' in extra) ? extra.sources : prev.sources,
        debug: (extra && 'debug' in extra) ? extra.debug : prev.debug,
        meta: (extra && 'meta' in extra) ? extra.meta : prev.meta,
      };
      this._renderMessages();
    }

    _renderMessages() {
      if (!this._els.body) return;
      this._els.body.innerHTML = '';

      for (const m of this._messages) {
        const msg = document.createElement('div');
        msg.className = `msg ${m.role}`;

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        if (m.role === 'assistant' && m.pending) {
          const sp = document.createElement('span');
          sp.className = 'spinner';
          bubble.appendChild(sp);
        } else {
          bubble.textContent = m.text || '';
        }

        msg.appendChild(bubble);

        if (m.role === 'assistant' && !m.pending) {
          const hasSources = this._cfg.enableCitations && Array.isArray(m.sources) && m.sources.length > 0;
          const showDebug = this._cfg.enableDebug && this._cfg.mode === 'admin' && m.debug;

          const meta = document.createElement('div');
          meta.className = 'meta';

          if (m.meta && typeof m.meta.latency_ms === 'number') {
            meta.appendChild(document.createTextNode(`Latency: ${m.meta.latency_ms}ms`));
          }
          if (m.meta && m.meta.cached) {
            const b = document.createElement('span');
            b.textContent = 'cached';
            meta.appendChild(b);
          }

          if (meta.childNodes.length) msg.appendChild(meta);

          if (hasSources) {
            const ul = document.createElement('ul');
            ul.className = 'citations';
            for (const s of m.sources) {
              const li = document.createElement('li');
              const a = document.createElement('a');
              a.href = s.url || '#';
              a.target = '_blank';
              a.rel = 'noopener noreferrer';
              a.textContent = s.title || s.url || 'source';
              li.appendChild(a);
              ul.appendChild(li);
            }
            msg.appendChild(ul);
          }

          if (showDebug) {
            const debugWrap = document.createElement('div');
            debugWrap.className = 'debugWrap';

            const toggle = document.createElement('button');
            toggle.className = 'debugToggle';
            toggle.textContent = 'Debug';

            const pre = document.createElement('pre');
            pre.textContent = safeJson(m.debug);
            pre.style.display = 'none';

            toggle.addEventListener('click', () => {
              const isOpen = pre.style.display !== 'none';
              pre.style.display = isOpen ? 'none' : 'block';
            });

            debugWrap.appendChild(toggle);
            debugWrap.appendChild(pre);
            msg.appendChild(debugWrap);
          }
        }

        this._els.body.appendChild(msg);
      }

      // Scroll to bottom
      this._els.body.scrollTop = this._els.body.scrollHeight;
    }
  }

  // Register custom element once
  if (!customElements.get('tmu-chat-widget')) {
    customElements.define('tmu-chat-widget', TMUChatWidget);
  }

  function resolveContainer(container) {
    if (!container) return null;
    if (typeof container === 'string') return document.querySelector(container);
    if (container instanceof HTMLElement) return container;
    return null;
  }

  // Public init API
  window.TMUChatbot = window.TMUChatbot || {};
  window.TMUChatbot.init = function init(options) {
    const cfg = { ...DEFAULTS, ...(options || {}) };
    cfg.apiBaseUrl = normalizeBaseUrl(cfg.apiBaseUrl);
    cfg.mode = cfg.mode === 'admin' ? 'admin' : 'public';

    const containerEl = resolveContainer(cfg.container) || document.body;

    const el = document.createElement('tmu-chat-widget');
    // Configure via properties rather than attributes for richer config
    el.setConfig({
      apiBaseUrl: cfg.apiBaseUrl,
      mode: cfg.mode,
      title: cfg.title,
      initialPrompt: cfg.initialPrompt,
      enableCitations: !!cfg.enableCitations,
      enableDebug: !!cfg.enableDebug,
      defaultParams: cfg.defaultParams || {},
      launcherIconUrl: cfg.launcherIconUrl || "",
      display: cfg.display === 'inline' ? 'inline' : 'floating',
    });

    containerEl.appendChild(el);
    return el;
  };
})();
