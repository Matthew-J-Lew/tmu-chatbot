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
      title: 'TMU Faculty of Arts Chatbot',
      enableCitations: true
    });
  </script>
*/

(function () {
  'use strict';

  const DEFAULTS = {
    apiBaseUrl: '',
    mode: 'public', // 'public' | 'admin'
    title: 'TMU Faculty of Arts Chatbot',
    initialPrompt: '',
    enableCitations: true,
    enableDebug: false,
    defaultParams: {},
    display: 'floating', // 'floating' | 'inline'
    launcherIconUrl: '',
    container: null,
  };

  function normalizeBaseUrl(url) {
    if (!url) return '';
    return String(url).replace(/\/+$/, '');
  }

  function detectAssetBase() {
    try {
      const cs = document.currentScript;
      if (cs && cs.src) return cs.src.replace(/\/[^\/]+$/, '/');
    } catch (_e) {}
    try {
      const scripts = Array.from(document.getElementsByTagName('script'));
      const hit = scripts.find((sc) => sc && sc.src && /\/widget\.js(\?|#|$)/.test(sc.src));
      if (hit && hit.src) return hit.src.replace(/\/[^\/]+$/, '/');
    } catch (_e) {}
    return '';
  }

  const ASSET_BASE = detectAssetBase();
  const DEFAULT_LAUNCHER_ICON_URL = ASSET_BASE ? `${ASSET_BASE}tmu_logo.png` : 'tmu_logo.png';

  function parseBool(val, fallback) {
    if (val === undefined || val === null || val === '') return fallback;
    const s = String(val).trim().toLowerCase();
    return s === '1' || s === 'true' || s === 'yes' || s === 'on';
  }

  function uuid() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
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

  function formatTime(ts) {
    try {
      return new Intl.DateTimeFormat([], {
        hour: 'numeric',
        minute: '2-digit',
      }).format(new Date(ts)).toLowerCase();
    } catch (_e) {
      return '';
    }
  }

  function formatDayLabel(ts) {
    const target = new Date(ts);
    const now = new Date();

    const startOf = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    const oneDay = 24 * 60 * 60 * 1000;
    const diff = Math.round((startOf(now) - startOf(target)) / oneDay);

    if (diff === 0) return 'Today';
    if (diff === 1) return 'Yesterday';

    try {
      return new Intl.DateTimeFormat([], {
        month: 'short',
        day: 'numeric',
        year: now.getFullYear() === target.getFullYear() ? undefined : 'numeric',
      }).format(target);
    } catch (_e) {
      return '';
    }
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_e) {
      data = { raw: text };
    }

    if (!res.ok) {
      const detail = data && (data.detail || data.message) ? (data.detail || data.message) : text;
      throw new Error(detail || `HTTP ${res.status}`);
    }

    return data;
  }

  async function sendMessage(apiBaseUrl, mode, question, sessionId, options) {
    const base = normalizeBaseUrl(apiBaseUrl);
    if (!base) throw new Error('Missing apiBaseUrl');

    const endpoint = mode === 'admin' ? '/admin/tools/chat' : '/api/chat';
    const url = base + endpoint;

    const body = mode === 'admin'
      ? {
          question,
          session_id: sessionId || undefined,
          params: options && options.params ? options.params : undefined,
        }
      : {
          question,
          session_id: sessionId || undefined,
        };

    return postJson(url, body);
  }

  function isMarkdownBlockStart(line) {
    const trimmed = String(line || '').trim();
    return (
      /^#{1,6}\s+/.test(trimmed) ||
      isHorizontalRule(trimmed) ||
      /^\s*[-+*]\s+/.test(line || '') ||
      /^\s*\d+\.\s+/.test(line || '')
    );
  }

  function isHorizontalRule(line) {
    const compact = String(line || '').trim().replace(/\s+/g, '');
    return compact.length >= 3 && (/^-+$/.test(compact) || /^\*+$/.test(compact) || /^_+$/.test(compact));
  }

  function escapeHtml(text) {
    return String(text || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function safeHref(href) {
    const raw = String(href || '').trim();
    if (!raw) return '';
    try {
      const parsed = new URL(raw, window.location.href);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return '';
      return parsed.href;
    } catch (_e) {
      return '';
    }
  }

  function renderInlineMarkdownHtml(text) {
    const placeholders = [];
    let html = escapeHtml(String(text || '').replace(/\u00a0/g, ' '));

    function store(fragment) {
      const key = `__MDTOKEN_${placeholders.length}__`;
      placeholders.push(fragment);
      return key;
    }

    html = html.replace(/`([^`]+)`/g, (_m, codeText) => store(`<code>${codeText}</code>`));

    html = html.replace(/\[([^\]]+)\]\(([^\s)]+)\)/g, (_m, label, href) => {
      const safe = safeHref(href);
      if (!safe) return label;
      return store(`<a href="${escapeHtml(safe)}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    });

    html = html.replace(/(?:\[(?:\d+(?:\s*,\s*\d+)*)\])+/g, (match) => {
      const ids = Array.from(new Set((match.match(/\d+/g) || []).map((value) => Number(value)).filter(Boolean)));
      if (!ids.length) return match;
      const pills = ids
        .map((id) => `<button type="button" class="citePill" data-cite-id="${id}" aria-label="Open source ${id}">[${id}]</button>`)
        .join('');
      return store(`<span class="citeGroup">${pills}</span>`);
    });

    html = html.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__([^_]+?)__/g, '<strong>$1</strong>');
    html = html.replace(/(^|[^*])\*([^*]+?)\*(?!\*)/g, '$1<em>$2</em>');
    html = html.replace(/(^|[^_])_([^_]+?)_(?!_)/g, '$1<em>$2</em>');

    return html.replace(/__MDTOKEN_(\d+)__/g, (_m, idx) => placeholders[Number(idx)] || '');
  }

  function parseListMarker(rawLine) {
    const match = /^(\s*)([-+*]|\d+\.)\s+(.+)$/.exec(rawLine || '');
    if (!match) return null;
    const indent = match[1].replace(/\t/g, '    ').length;
    const marker = match[2];
    return {
      indent,
      type: /\d+\./.test(marker) ? 'ol' : 'ul',
      content: match[3],
    };
  }

  function buildListHtml(lines, startIndex) {
    function nextNonEmptyIndex(fromIndex) {
      for (let idx = fromIndex; idx < lines.length; idx += 1) {
        if (String(lines[idx] || '').trim()) return idx;
      }
      return -1;
    }

    function renderContinuationHtml(parts) {
      const filtered = parts.filter((part) => part !== null && part !== undefined && String(part) !== '');
      if (!filtered.length) return '';
      return `<div class="mdListContinuation">${filtered.join('<br>')}</div>`;
    }

    function parseListBlock(index, baseIndent) {
      const first = parseListMarker(lines[index]);
      if (!first) return { html: '', nextIndex: index };

      const listType = first.type;
      let html = `<${listType} class="mdList md${listType === 'ul' ? 'Ul' : 'Ol'}">`;
      let i = index;

      while (i < lines.length) {
        while (i < lines.length && !String(lines[i] || '').trim()) {
          const lookahead = nextNonEmptyIndex(i + 1);
          if (lookahead < 0) {
            i = lines.length;
            break;
          }
          const nextMarker = parseListMarker(lines[lookahead]);
          if (!nextMarker || nextMarker.indent < baseIndent) break;
          i = lookahead;
        }

        if (i >= lines.length) break;

        const marker = parseListMarker(lines[i]);
        if (!marker || marker.indent !== baseIndent || marker.type !== listType) break;

        let itemHtml = renderInlineMarkdownHtml(marker.content.trim());
        i += 1;

        const continuationParts = [];

        while (i < lines.length) {
          const currentLine = lines[i];
          const trimmed = String(currentLine || '').trim();

          if (!trimmed) {
            const lookahead = nextNonEmptyIndex(i + 1);
            if (lookahead < 0) {
              i = lines.length;
              break;
            }
            const nextMarker = parseListMarker(lines[lookahead]);
            if (nextMarker) {
              if (nextMarker.indent > baseIndent) {
                i = lookahead;
                continue;
              }
              if (nextMarker.indent <= baseIndent) {
                i = lookahead;
                break;
              }
            }
            continuationParts.push('');
            i = lookahead;
            continue;
          }

          const nextMarker = parseListMarker(currentLine);
          if (nextMarker) {
            if (nextMarker.indent > baseIndent) {
              itemHtml += renderContinuationHtml(continuationParts);
              continuationParts.length = 0;
              const nested = parseListBlock(i, nextMarker.indent);
              itemHtml += nested.html;
              i = nested.nextIndex;
              continue;
            }
            if (nextMarker.indent <= baseIndent) {
              break;
            }
          }

          continuationParts.push(renderInlineMarkdownHtml(trimmed));
          i += 1;
        }

        itemHtml += renderContinuationHtml(continuationParts);
        html += `<li>${itemHtml}</li>`;
      }

      html += `</${listType}>`;
      return { html, nextIndex: i };
    }

    const firstItem = parseListMarker(lines[startIndex]);
    if (!firstItem) return { html: '', nextIndex: startIndex + 1 };
    return parseListBlock(startIndex, firstItem.indent);
  }

  function renderAssistantMarkdown(text) {
    const normalized = String(text || '')
      .replace(/\r\n?/g, '\n')
      .replace(/\u200b/g, '')
      .replace(/\u00a0/g, ' ');
    const lines = normalized.split('\n');
    let html = '';
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = String(line || '').trim();

      if (!trimmed) {
        i += 1;
        continue;
      }

      const headingMatch = /^(#{1,6})\s+(.+)$/.exec(trimmed);
      if (headingMatch) {
        const level = Math.min(headingMatch[1].length, 4);
        html += `<h${level} class="mdHeading mdHeading${level}">${renderInlineMarkdownHtml(headingMatch[2].trim())}</h${level}>`;
        i += 1;
        continue;
      }

      if (isHorizontalRule(trimmed)) {
        html += '<hr class="mdRule">';
        i += 1;
        continue;
      }

      if (parseListMarker(line)) {
        const built = buildListHtml(lines, i);
        html += built.html;
        i = built.nextIndex;
        continue;
      }

      const paraLines = [];
      while (i < lines.length) {
        const candidate = lines[i];
        if (!String(candidate || '').trim()) break;
        if (paraLines.length && isMarkdownBlockStart(candidate)) break;
        if (!paraLines.length && (isHorizontalRule(String(candidate || '').trim()) || /^(#{1,6})\s+/.test(String(candidate || '').trim()) || parseListMarker(candidate))) break;
        paraLines.push(String(candidate || '').trimRight());
        i += 1;
      }

      if (paraLines.length) {
        html += `<p class="mdParagraph">${paraLines.map((part) => renderInlineMarkdownHtml(part)).join('<br>')}</p>`;
      } else {
        i += 1;
      }
    }

    const template = document.createElement('template');
    template.innerHTML = `<div class="bubbleContent">${html}</div>`;

    for (const a of template.content.querySelectorAll('a')) {
      const href = safeHref(a.getAttribute('href'));
      if (!href) {
        a.replaceWith(document.createTextNode(a.textContent || ''));
        continue;
      }
      a.setAttribute('href', href);
      a.setAttribute('target', '_blank');
      a.setAttribute('rel', 'noopener noreferrer');
    }

    return template.content;
  }

  function displaySourceTitle(source) {
    const title = source && source.title ? String(source.title).trim() : '';
    if (title && !/^https?:\/\//i.test(title)) return title;

    const rawUrl = source && source.url ? String(source.url) : '';
    if (!rawUrl) return 'Official TMU source';
    try {
      const parsed = new URL(rawUrl, window.location.href);
      const tail = parsed.pathname.replace(/\/+$/, '').split('/').filter(Boolean).pop();
      if (!tail) return parsed.hostname;
      const pretty = tail.replace(/[-_]+/g, ' ').trim();
      return pretty ? pretty.charAt(0).toUpperCase() + pretty.slice(1) : parsed.hostname;
    } catch (_e) {
      return rawUrl;
    }
  }

  function normalizeCompareText(value) {
    return String(value || '')
      .toLowerCase()
      .replace(/&/g, ' and ')
      .replace(/[^a-z0-9]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function sourceContextLabel(source) {
    const rawUrl = source && source.url ? String(source.url) : '';
    if (!rawUrl) return '';
    try {
      const parsed = new URL(rawUrl, window.location.href);
      const path = parsed.pathname.toLowerCase();
      if (path.includes('/calendar/')) return 'TMU Calendar';
      if (path.includes('/myservicehub-support/')) return 'MyServiceHub Support';
      if (path.includes('/current-students/')) return 'Current Students';
      if (path.includes('/student-wellbeing/')) return 'Student Wellbeing';
      if (path.includes('/arts/')) return 'Faculty of Arts';
      return parsed.hostname.replace(/^www\./i, '');
    } catch (_e) {
      return '';
    }
  }

  function displaySourceSubtitle(source) {
    const title = displaySourceTitle(source);
    const titleNorm = normalizeCompareText(title);
    const section = source && source.section ? String(source.section).trim() : '';
    const sectionNorm = normalizeCompareText(section);
    if (section && sectionNorm && !titleNorm.includes(sectionNorm) && !sectionNorm.includes(titleNorm)) {
      return section;
    }
    const label = sourceContextLabel(source);
    const labelNorm = normalizeCompareText(label);
    if (label && labelNorm && !titleNorm.includes(labelNorm) && !labelNorm.includes(titleNorm)) {
      return label;
    }
    return '';
  }

  function buildSourcePanel(sources) {
    const wrap = document.createElement('div');
    wrap.className = 'sourceWrap';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'sourceToggle';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.textContent = `Sources (${sources.length})`;

    const list = document.createElement('div');
    list.className = 'sourceList';
    list.hidden = true;

    for (const source of sources) {
      const item = document.createElement('div');
      item.className = 'sourceItem';
      item.dataset.sourceId = String(source.id);

      const badge = document.createElement('span');
      badge.className = 'sourceBadge';
      badge.textContent = `[${source.id}]`;
      item.appendChild(badge);

      const body = document.createElement('div');
      body.className = 'sourceBody';

      const link = document.createElement('a');
      const safeUrl = safeHref(source.url);
      link.className = 'sourceTitle';
      link.href = safeUrl || '#';
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = displaySourceTitle(source);
      body.appendChild(link);

      const subtitleText = displaySourceSubtitle(source);
      if (subtitleText) {
        const subtitle = document.createElement('div');
        subtitle.className = 'sourceSubtitle';
        subtitle.textContent = subtitleText;
        body.appendChild(subtitle);
      }

      item.appendChild(body);
      list.appendChild(item);
    }

    function setOpen(open) {
      list.hidden = !open;
      wrap.classList.toggle('open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    wrap.openSource = function (sourceId) {
      if (!sourceId) return;
      setOpen(true);
      const target = list.querySelector(`.sourceItem[data-source-id="${String(sourceId)}"]`);
      if (!target) return;
      target.classList.remove('flash');
      void target.offsetWidth;
      target.classList.add('flash');
      target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    };

    toggle.addEventListener('click', () => setOpen(list.hidden));

    wrap.appendChild(toggle);
    wrap.appendChild(list);
    return wrap;
  }

  const STYLE = `
    :host {
      display: block;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #18181b;
    }

    * { box-sizing: border-box; }

    .root {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 2147483647;
    }

    .root.inline {
      position: relative;
      right: auto;
      bottom: auto;
    }

    .launcher {
      width: 58px;
      height: 58px;
      border-radius: 999px;
      border: none;
      background: #5e93eb;
      color: #fff;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.28);
      cursor: pointer;
      padding: 0;
      user-select: none;
      transition: transform 180ms ease, box-shadow 180ms ease, opacity 180ms ease;
    }

    .launcher:hover {
      transform: translateY(-2px) scale(1.02);
      box-shadow: 0 20px 38px rgba(15, 23, 42, 0.33);
      background: #4f86e2;
    }

    .launcher:active {
      transform: translateY(0) scale(0.98);
    }

    .launcherImg {
      width: 32px;
      height: 32px;
      object-fit: contain;
      display: block;
    }

    .panel {
      position: absolute;
      right: 0;
      bottom: 72px;
      width: min(376px, calc(100vw - 20px));
      height: min(640px, calc(100vh - 24px));
      background: #fff;
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 22px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 24px 55px rgba(15, 23, 42, 0.24);
      transform-origin: bottom right;
      transition: opacity 220ms ease, transform 220ms ease, visibility 220ms ease;
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      transform: translateY(18px) scale(0.985);
    }

    .panel.open {
      opacity: 1;
      visibility: visible;
      pointer-events: auto;
      transform: translateY(0) scale(1);
    }

    .root.inline .panel {
      position: static;
      width: 100%;
      height: 640px;
      max-height: 80vh;
      opacity: 1;
      visibility: visible;
      pointer-events: auto;
      transform: none;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    }

    .root.inline .launcher {
      display: none;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      background: #5e93eb;
      color: #fff;
      border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }

    .brand {
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brandIcon {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.88);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      overflow: hidden;
    }

    .brandIcon img {
      width: 22px;
      height: 22px;
      object-fit: contain;
      display: block;
    }

    .titleWrap {
      min-width: 0;
    }

    .title {
      font-size: 14px;
      font-weight: 700;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .subtitle {
      font-size: 11px;
      opacity: 0.92;
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 6px;
      flex: 0 0 auto;
    }

    button {
      font: inherit;
      cursor: pointer;
      border: 0;
      background: transparent;
      color: inherit;
    }

    .iconBtn {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      background: rgba(255, 255, 255, 0.12);
      transition: background 160ms ease, transform 160ms ease, opacity 160ms ease;
      padding: 0;
      line-height: 1;
      font-size: 15px;
    }

    .iconBtn:hover {
      background: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }

    .iconBtn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
    }

    .closeGlyph {
      font-size: 18px;
      transform: none;
      line-height: 1;
    }

    .body {
      flex: 1;
      overflow: auto;
      padding: 0;
      background: #fff;
      scroll-behavior: smooth;
    }

    .scrollInner {
      min-height: 100%;
      display: flex;
      flex-direction: column;
      padding: 18px 14px 16px;
      gap: 10px;
    }

    .welcome {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      padding: 18px 10px 20px;
      gap: 10px;
      color: #27272a;
    }

    .welcomeLogoWrap {
      width: 72px;
      height: 72px;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: #f5f5f5;
      border: 1px solid rgba(15, 23, 42, 0.06);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }

    .welcomeLogoWrap img {
      width: 42px;
      height: 42px;
      object-fit: contain;
      display: block;
    }

    .welcomeTitle {
      font-size: 18px;
      font-weight: 700;
      line-height: 1.25;
      max-width: 280px;
    }

    .welcomeText {
      font-size: 13px;
      line-height: 1.45;
      color: #71717a;
      max-width: 280px;
    }

    .dayDivider {
      align-self: center;
      margin: 2px 0 4px;
      font-size: 12px;
      color: #71717a;
    }

    .msg {
      display: flex;
      flex-direction: column;
      gap: 4px;
      position: relative;
    }

    .msg.enter {
      animation: msgIn 240ms ease both;
    }

    @keyframes msgIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .user { align-items: flex-end; }
    .assistant { align-items: flex-start; }

    .bubble {
      position: relative;
      max-width: 88%;
      padding: 11px 13px;
      border-radius: 14px;
      font-size: 14px;
      line-height: 1.42;
      white-space: normal;
      word-break: break-word;
      transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, border-color 180ms ease;
    }

    .assistant .bubble {
      background: #f4f4f5;
      border: 1px solid #ececf0;
      color: #18181b;
      border-bottom-left-radius: 8px;
    }

    .user .bubble {
      background: #e9f0ff;
      border: 1px solid #d3e0ff;
      color: #16325f;
      border-bottom-right-radius: 8px;
      white-space: pre-wrap;
    }

    .msg:hover .bubble {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }

    .timestamp {
      font-size: 11px;
      color: #71717a;
      opacity: 0;
      transform: translateY(3px);
      transition: opacity 160ms ease, transform 160ms ease;
      padding: 0 4px;
      pointer-events: none;
    }

    .assistant .timestamp { align-self: flex-start; }
    .user .timestamp { align-self: flex-end; }
    .msg:hover .timestamp { opacity: 1; transform: translateY(0); }

    .meta {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: #71717a;
      padding: 0 4px;
    }

    .bubbleContent > :first-child { margin-top: 0; }
    .bubbleContent > :last-child { margin-bottom: 0; }

    .mdHeading {
      margin: 0 0 6px;
      line-height: 1.25;
      font-weight: 700;
      color: inherit;
    }

    .mdHeading1 { font-size: 16px; }
    .mdHeading2, .mdHeading3 { font-size: 15px; }

    .mdParagraph {
      margin: 0 0 6px;
    }

    .mdRule {
      border: 0;
      border-top: 1px solid #d4d4d8;
      margin: 8px 0;
    }

    .mdList {
      margin: 0 0 6px;
      padding-left: 18px;
    }

    .mdList .mdList {
      margin-top: 2px;
      margin-bottom: 2px;
    }

    .mdList li {
      margin: 2px 0;
    }

    .bubbleContent code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
      background: rgba(24, 24, 27, 0.06);
      border: 1px solid rgba(24, 24, 27, 0.08);
      border-radius: 6px;
      padding: 1px 4px;
    }

    .assistant .bubble a {
      color: #1d4a97;
      text-decoration: none;
    }

    .assistant .bubble a:hover {
      text-decoration: underline;
    }

    .citeGroup {
      display: inline-flex;
      gap: 4px;
      flex-wrap: wrap;
      margin-left: 2px;
      vertical-align: baseline;
    }

    .citePill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 24px;
      padding: 1px 6px;
      border-radius: 999px;
      border: 1px solid rgba(29, 74, 151, 0.18);
      background: rgba(255, 255, 255, 0.85);
      color: #1d4a97;
      font-size: 11px;
      font-weight: 600;
      line-height: 1.45;
    }

    .assistant .bubble .citePill:hover {
      background: #eef4ff;
      text-decoration: none;
    }

    .sourceWrap {
      width: min(88%, 100%);
      margin-top: 4px;
    }

    .sourceToggle {
      font-size: 11px;
      padding: 5px 9px;
      border-radius: 999px;
      background: #f4f4f5;
      color: #3f3f46;
      border: 1px solid #e4e4e7;
    }

    .sourceList {
      margin-top: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .sourceItem {
      display: flex;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid #ececf0;
      background: #fafafa;
      align-items: flex-start;
    }

    .sourceItem.flash {
      animation: sourceFlash 1000ms ease;
    }

    @keyframes sourceFlash {
      0% { background: #eef4ff; border-color: #bfd3ff; }
      100% { background: #fafafa; border-color: #ececf0; }
    }

    .sourceBadge {
      flex: 0 0 auto;
      color: #1d4a97;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.4;
      margin-top: 1px;
    }

    .sourceBody {
      min-width: 0;
    }

    .sourceTitle {
      color: #1d4a97;
      text-decoration: none;
      font-size: 12px;
      font-weight: 600;
      line-height: 1.35;
    }

    .sourceTitle:hover {
      text-decoration: underline;
    }

    .sourceSubtitle {
      margin-top: 2px;
      color: #71717a;
      font-size: 11px;
      line-height: 1.35;
    }

    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 2px solid rgba(29, 74, 151, 0.2);
      border-top-color: rgba(29, 74, 151, 0.9);
      animation: spin 900ms linear infinite;
      vertical-align: middle;
    }

    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }

    .debugWrap {
      width: min(88%, 100%);
      margin-top: 2px;
    }

    .debugToggle {
      font-size: 11px;
      padding: 5px 9px;
      border-radius: 999px;
      background: #f4f4f5;
      color: #3f3f46;
      border: 1px solid #e4e4e7;
    }

    pre {
      margin: 8px 0 0 0;
      padding: 10px 12px;
      border-radius: 12px;
      background: #fafafa;
      border: 1px solid #e4e4e7;
      overflow: auto;
      font-size: 11px;
      color: #27272a;
      max-width: 100%;
    }

    .footer {
      padding: 12px 14px 14px;
      border-top: 1px solid rgba(15, 23, 42, 0.08);
      background: #fff;
    }

    .row {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    input[type="text"] {
      flex: 1;
      min-width: 0;
      font: inherit;
      border-radius: 12px;
      border: 1px solid #d4d4d8;
      padding: 12px 14px;
      font-size: 14px;
      outline: none;
      transition: border-color 160ms ease, box-shadow 160ms ease;
      background: #fff;
      color: #18181b;
    }

    input[type="text"]::placeholder {
      color: #a1a1aa;
    }

    input[type="text"]:focus {
      border-color: rgba(29, 74, 151, 0.45);
      box-shadow: 0 0 0 3px rgba(29, 74, 151, 0.1);
    }

    .sendBtn {
      flex: 0 0 auto;
      border-radius: 12px;
      padding: 12px 14px;
      background: #5e93eb;
      color: #fff;
      font-size: 13px;
      font-weight: 600;
      min-width: 64px;
      transition: transform 160ms ease, background 160ms ease, opacity 160ms ease;
    }

    .sendBtn:hover {
      background: #4f86e2;
      transform: translateY(-1px);
    }

    .sendBtn:active {
      transform: translateY(0);
    }

    .sendBtn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .hint {
      margin-top: 8px;
      font-size: 11px;
      color: #71717a;
      line-height: 1.35;
      min-height: 15px;
    }

    @media (max-width: 480px) {
      .root {
        right: 10px;
        bottom: 10px;
      }

      .panel {
        width: min(100vw - 12px, 376px);
        height: min(640px, calc(100vh - 12px));
        bottom: 68px;
      }
    }
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
      this._open = false;
      this._els = {};
    }

    connectedCallback() {
      this._sessionId = this._getOrCreateSessionId();
      this._readAttrsIntoConfig();
      this._render();
      this._ensureWelcomeMessage();
    }

    attributeChangedCallback() {
      this._readAttrsIntoConfig();
      this._updateHeader();
      this._updatePanelVisibility();
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
      if (initialPrompt !== null) this._cfg.initialPrompt = initialPrompt || '';
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
      launcher.addEventListener('click', () => this.toggle());

      const panel = document.createElement('div');
      panel.className = 'panel';

      const header = document.createElement('div');
      header.className = 'header';

      const brand = document.createElement('div');
      brand.className = 'brand';

      const brandIcon = document.createElement('div');
      brandIcon.className = 'brandIcon';
      const brandImg = document.createElement('img');
      brandImg.alt = 'TMU';
      brandIcon.appendChild(brandImg);

      const titleWrap = document.createElement('div');
      titleWrap.className = 'titleWrap';

      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = this._cfg.title;

      const subtitle = document.createElement('div');
      subtitle.className = 'subtitle';
      subtitle.textContent = this._cfg.mode === 'admin' ? 'Admin mode enabled' : 'Programs, courses, requirements';

      titleWrap.appendChild(title);
      titleWrap.appendChild(subtitle);
      brand.appendChild(brandIcon);
      brand.appendChild(titleWrap);

      const actions = document.createElement('div');
      actions.className = 'actions';

      const btnCopy = this._createIconButton('⎘', 'Copy last answer');
      btnCopy.addEventListener('click', () => this.copyLastAnswer());

      const btnReset = this._createIconButton('↺', 'Reset conversation');
      btnReset.addEventListener('click', () => this.reset());

      const btnClose = this._createIconButton('×', 'Close chat');
      btnClose.classList.add('closeGlyph');
      btnClose.addEventListener('click', () => this.close());

      actions.appendChild(btnCopy);
      actions.appendChild(btnReset);
      actions.appendChild(btnClose);

      header.appendChild(brand);
      header.appendChild(actions);

      const body = document.createElement('div');
      body.className = 'body';

      const footer = document.createElement('div');
      footer.className = 'footer';

      const row = document.createElement('div');
      row.className = 'row';

      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Message...';
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.send();
        }
      });

      const btnSend = document.createElement('button');
      btnSend.className = 'sendBtn';
      btnSend.textContent = 'Send';
      btnSend.addEventListener('click', () => this.send());

      row.appendChild(input);
      row.appendChild(btnSend);

      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = this._cfg.mode === 'admin' ? 'Admin mode: debug details may be available.' : '';

      footer.appendChild(row);
      footer.appendChild(hint);

      panel.appendChild(header);
      panel.appendChild(body);
      panel.appendChild(footer);

      root.appendChild(panel);
      root.appendChild(launcher);

      this._shadow.appendChild(style);
      this._shadow.appendChild(root);

      this._els = {
        root,
        launcher,
        panel,
        header,
        title,
        subtitle,
        brandImg,
        body,
        footer,
        input,
        btnSend,
        btnCopy,
        btnReset,
        btnClose,
        hint,
      };

      this._launcherIconUrl = this._cfg.launcherIconUrl || DEFAULT_LAUNCHER_ICON_URL;
      this._setLauncherIcon(this._open);
      this._updateBrandImages();
      this._renderMessages();
      this._updateHeader();
      this._updatePanelVisibility();
    }

    _createIconButton(text, label) {
      const btn = document.createElement('button');
      btn.className = 'iconBtn';
      btn.type = 'button';
      btn.textContent = text;
      btn.setAttribute('aria-label', label);
      btn.title = label;
      return btn;
    }

    _updateBrandImages() {
      if (this._els.brandImg) {
        this._els.brandImg.src = this._launcherIconUrl || DEFAULT_LAUNCHER_ICON_URL;
      }
    }

    _getWelcomePrompt() {
      return this._cfg.initialPrompt || 'Hello! Welcome to the Faculty of Arts. What can I help you with today?';
    }

    _ensureWelcomeMessage() {
      if (this._messages.length > 0) return;
      this._appendAssistant(this._getWelcomePrompt(), { sources: null, debug: null, meta: null });
    }

    _setLauncherIcon(isOpen) {
      if (!this._els.launcher) return;
      const btn = this._els.launcher;
      while (btn.firstChild) btn.removeChild(btn.firstChild);

      if (isOpen) {
        const span = document.createElement('span');
        span.textContent = '×';
        span.style.fontSize = '22px';
        span.style.lineHeight = '1';
        span.style.transform = 'none';
        span.style.display = 'block';
        btn.appendChild(span);
      } else {
        const img = document.createElement('img');
        img.className = 'launcherImg';
        img.alt = 'TMU';
        img.src = this._launcherIconUrl || DEFAULT_LAUNCHER_ICON_URL;
        btn.appendChild(img);
      }
    }

    _updateHeader() {
      if (this._els.title) this._els.title.textContent = this._cfg.title;
      if (this._els.subtitle) {
        this._els.subtitle.textContent = this._cfg.mode === 'admin'
          ? 'Admin mode enabled'
          : 'Programs, courses, requirements';
      }
      if (this._els.hint) {
        this._els.hint.textContent = this._cfg.mode === 'admin'
          ? 'Admin mode: debug details may be available.'
          : '';
      }
      if (this._els.btnClose) {
        this._els.btnClose.style.display = this._cfg.display === 'inline' ? 'none' : 'inline-flex';
      }
      this._updateBrandImages();
    }

    _updatePanelVisibility() {
      if (!this._els.panel || !this._els.launcher || !this._els.root) return;
      const inline = this._cfg.display === 'inline';
      this._els.root.className = `root ${inline ? 'inline' : ''}`.trim();

      if (inline) {
        this._els.panel.classList.add('open');
        this._els.launcher.setAttribute('aria-hidden', 'true');
        return;
      }

      this._els.panel.classList.toggle('open', !!this._open);
      this._els.launcher.setAttribute('aria-label', this._open ? 'Close chat' : 'Open chat');
      this._setLauncherIcon(this._open);
    }

    open() {
      if (this._cfg.display === 'inline') return;
      this._open = true;
      this._updatePanelVisibility();
      setTimeout(() => {
        try {
          if (this._els.input) this._els.input.focus();
        } catch (_e) {}
      }, 120);
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
      this._ensureWelcomeMessage();
      if (this._els.input) this._els.input.focus();
    }

    async copyLastAnswer() {
      const last = [...this._messages].reverse().find((m) => m.role === 'assistant' && m.text);
      if (!last) return;

      try {
        await navigator.clipboard.writeText(last.text);
      } catch (_e) {
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

      const placeholderId = uuid();
      this._appendAssistant('', { pending: true, id: placeholderId, sources: null, debug: null, meta: null });

      this._sending = true;
      this._setSendDisabled(true);

      try {
        const params = this._cfg.defaultParams && typeof this._cfg.defaultParams === 'object'
          ? this._cfg.defaultParams
          : {};

        const resp = await sendMessage(
          this._cfg.apiBaseUrl,
          this._cfg.mode,
          question,
          this._sessionId,
          { params }
        );

        const answer = resp && resp.answer ? String(resp.answer) : '';
        const sources = resp && resp.sources ? resp.sources : null;
        const debug = resp && resp.debug ? resp.debug : null;
        const meta = {
          latency_ms: resp && resp.latency_ms,
          cached: resp && resp.cached,
          timings: resp && resp.timings,
        };

        this._replaceAssistant(placeholderId, answer, { sources, debug, meta, reanimate: true });
      } catch (err) {
        const msg = err && err.message ? err.message : 'Request failed.';
        this._replaceAssistant(
          placeholderId,
          `Sorry — I couldn't complete that request.\n\n${msg}`,
          { sources: null, debug: null, meta: null, reanimate: true }
        );
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
      this._messages.push({
        role: 'user',
        text: String(text),
        createdAt: Date.now(),
        _rendered: false,
      });
      this._renderMessages();
    }

    _appendAssistant(text, extra) {
      const m = {
        role: 'assistant',
        text: String(text || ''),
        pending: !!(extra && extra.pending),
        id: extra && extra.id ? extra.id : uuid(),
        sources: extra && 'sources' in extra ? extra.sources : null,
        debug: extra && 'debug' in extra ? extra.debug : null,
        meta: extra && 'meta' in extra ? extra.meta : null,
        createdAt: extra && extra.createdAt ? extra.createdAt : Date.now(),
        _rendered: false,
      };
      this._messages.push(m);
      this._renderMessages();
      return m.id;
    }

    _replaceAssistant(id, text, extra) {
      const idx = this._messages.findIndex((m) => m.role === 'assistant' && m.id === id);
      if (idx === -1) return;
      const prev = this._messages[idx];
      this._messages[idx] = {
        ...prev,
        text: String(text || ''),
        pending: false,
        sources: extra && 'sources' in extra ? extra.sources : prev.sources,
        debug: extra && 'debug' in extra ? extra.debug : prev.debug,
        meta: extra && 'meta' in extra ? extra.meta : prev.meta,
        _rendered: !(extra && extra.reanimate),
      };
      this._renderMessages();
    }

    _renderMessages() {
      if (!this._els.body) return;
      this._els.body.innerHTML = '';

      const scrollInner = document.createElement('div');
      scrollInner.className = 'scrollInner';

      const welcome = document.createElement('div');
      welcome.className = 'welcome';

      const welcomeLogoWrap = document.createElement('div');
      welcomeLogoWrap.className = 'welcomeLogoWrap';
      const welcomeLogo = document.createElement('img');
      welcomeLogo.alt = 'TMU';
      welcomeLogo.src = this._launcherIconUrl || DEFAULT_LAUNCHER_ICON_URL;
      welcomeLogoWrap.appendChild(welcomeLogo);

      const welcomeTitle = document.createElement('div');
      welcomeTitle.className = 'welcomeTitle';
      welcomeTitle.textContent = this._cfg.title;

      const welcomeText = document.createElement('div');
      welcomeText.className = 'welcomeText';
      welcomeText.textContent = 'Our virtual assistant is here to help with Faculty of Arts questions.';

      welcome.appendChild(welcomeLogoWrap);
      welcome.appendChild(welcomeTitle);
      welcome.appendChild(welcomeText);
      scrollInner.appendChild(welcome);

      let lastDayLabel = null;

      for (const m of this._messages) {
        const dayLabel = formatDayLabel(m.createdAt || Date.now());
        if (dayLabel && dayLabel !== lastDayLabel) {
          const divider = document.createElement('div');
          divider.className = 'dayDivider';
          divider.textContent = dayLabel;
          scrollInner.appendChild(divider);
          lastDayLabel = dayLabel;
        }

        const msg = document.createElement('div');
        msg.className = `msg ${m.role}${m._rendered ? '' : ' enter'}`;
        m._rendered = true;

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        if (m.role === 'assistant' && m.pending) {
          const sp = document.createElement('span');
          sp.className = 'spinner';
          bubble.appendChild(sp);
        } else if (m.role === 'assistant') {
          bubble.appendChild(renderAssistantMarkdown(m.text || ''));
        } else {
          bubble.textContent = m.text || '';
        }

        msg.appendChild(bubble);

        const timestamp = document.createElement('div');
        timestamp.className = 'timestamp';
        timestamp.textContent = formatTime(m.createdAt || Date.now());
        msg.appendChild(timestamp);

        if (m.role === 'assistant' && !m.pending) {
          const hasSources = this._cfg.enableCitations && Array.isArray(m.sources) && m.sources.length > 0;
          const showDebug = this._cfg.enableDebug && this._cfg.mode === 'admin' && m.debug;

          const meta = document.createElement('div');
          meta.className = 'meta';

          if (m.meta && typeof m.meta.latency_ms === 'number') {
            meta.appendChild(document.createTextNode(`Latency: ${m.meta.latency_ms}ms`));
          }
          if (m.meta && m.meta.cached) {
            const cached = document.createElement('span');
            cached.textContent = 'cached';
            meta.appendChild(cached);
          }
          if (meta.childNodes.length) msg.appendChild(meta);

          if (hasSources) {
            const sourcePanel = buildSourcePanel(m.sources);
            msg.appendChild(sourcePanel);

            bubble.addEventListener('click', (event) => {
              const target = event.target && event.target.closest ? event.target.closest('.citePill') : null;
              if (!target) return;
              event.preventDefault();
              const sourceId = Number(target.getAttribute('data-cite-id'));
              if (!sourceId || typeof sourcePanel.openSource !== 'function') return;
              sourcePanel.openSource(sourceId);
            });
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

        scrollInner.appendChild(msg);
      }

      this._els.body.appendChild(scrollInner);
      this._els.body.scrollTop = this._els.body.scrollHeight;
    }
  }

  if (!customElements.get('tmu-chat-widget')) {
    customElements.define('tmu-chat-widget', TMUChatWidget);
  }

  function resolveContainer(container) {
    if (!container) return null;
    if (typeof container === 'string') return document.querySelector(container);
    if (container instanceof HTMLElement) return container;
    return null;
  }

  window.TMUChatbot = window.TMUChatbot || {};
  window.TMUChatbot.init = function init(options) {
    const cfg = { ...DEFAULTS, ...(options || {}) };
    cfg.apiBaseUrl = normalizeBaseUrl(cfg.apiBaseUrl);
    cfg.mode = cfg.mode === 'admin' ? 'admin' : 'public';

    const containerEl = resolveContainer(cfg.container) || document.body;

    const el = document.createElement('tmu-chat-widget');
    el.setConfig({
      apiBaseUrl: cfg.apiBaseUrl,
      mode: cfg.mode,
      title: cfg.title,
      initialPrompt: cfg.initialPrompt,
      enableCitations: !!cfg.enableCitations,
      enableDebug: !!cfg.enableDebug,
      defaultParams: cfg.defaultParams || {},
      launcherIconUrl: cfg.launcherIconUrl || '',
      display: cfg.display === 'inline' ? 'inline' : 'floating',
    });

    containerEl.appendChild(el);
    return el;
  };
})();
