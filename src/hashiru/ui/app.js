"use strict";
// The chat page. It is a plain client of this hub's OpenAI-compatible API (/v1/chat/completions, streamed), plus
// /v1/hashiru/... for the agents panel. Chats live in this browser's localStorage; the hub keeps the agents.
(function () {
  const $ = (id) => document.getElementById(id);
  const MAX_CHATS = 60;
  const EXAMPLES = [
    ["Say hello", "Hire one worker on a local model and ask it to say hello."],
    ["Research a topic", "Find three recent arXiv papers on budget-aware multi-agent systems and summarize each in two sentences."],
    ["Argue both sides", "Hire two workers, one to argue for remote work and one against, then summarize both sides."],
    ["Check the budget", "How much budget do you have, and which models can you hire?"],
  ];

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window or full: keep going */ }
    },
  };

  const state = {
    chats: Array.isArray(store.get("hashiru.chats", [])) ? store.get("hashiru.chats", []) : [],
    current: null,
    key: store.get("hashiru.key", ""),
    theme: store.get("hashiru.theme", "auto"),
    busy: false,
    controller: null,
    info: null,           // /v1/hashiru/info
    draftPrivate: false,  // the lock, before a new chat's first message
    draftMode: "act",     // the Plan switch, before a new chat's first message
    agentsTimer: 0,
  };

  // ---- helpers ----
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const usd = (n) => {
    if (n === undefined || n === null) return "–";
    const v = Number(n);
    return `$${v === 0 || Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toFixed(4)}`;
  };
  const headers = (extra) => Object.assign(state.key ? { Authorization: `Bearer ${state.key}` } : {}, extra || {});
  const chat = () => state.chats.find((c) => c.id === state.current) || null;
  const save = () => store.set("hashiru.chats", state.chats.slice(0, MAX_CHATS));

  async function api(path, options) {
    const res = await fetch(path, Object.assign({}, options, { headers: headers(options && options.headers) }));
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const err = new Error((body.error && body.error.message) || res.statusText || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res;
  }

  // ---- theme ----
  function applyTheme() {
    if (state.theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = state.theme;
    $("theme-label").textContent = `Theme: ${state.theme}`;
  }

  // ---- sidebar ----
  function renderChats() {
    const list = $("chats");
    list.replaceChildren();
    [...state.chats].sort((a, b) => b.updated - a.updated).forEach((c) => {
      const row = el("div", `chat-row${c.id === state.current ? " current" : ""}`);
      const open = el("button", "open", c.title || "New chat");
      if (c.private) open.insertAdjacentHTML("afterbegin", LOCK);
      open.type = "button";
      open.title = c.title || "";
      open.addEventListener("click", () => { openChat(c.id); closeSide(); });
      const del = el("button", "icon-btn del");
      del.type = "button";
      del.setAttribute("aria-label", `Delete chat: ${c.title}`);
      del.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>';
      del.addEventListener("click", () => deleteChat(c.id));
      row.append(open, del);
      list.append(row);
    });
  }

  // ---- private chats ----
  const LOCK = '<svg class="lk" viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>';
  const FINE = "Agents can make mistakes, and cloud models spend real money. Check the agents panel.";
  const isPrivate = () => { const c = chat(); return c ? Boolean(c.private) : state.draftPrivate; };

  // The lock, the header and the fine print. A chat that went private stays private (the hub keeps it so), so its
  // lock cannot be turned off. When the hub obfuscates, cloud models see a private chat with names and details hidden;
  // when it blocks, only local models see it.
  function renderPrivacy() {
    const on = isPrivate();
    const c = chat();
    const lock = $("private");
    const info = state.info;
    const obfuscating = Boolean(info && info.private_mode === "obfuscate");
    const noLocal = Boolean(info) && !obfuscating && !info.private_ceo_model;
    lock.setAttribute("aria-pressed", String(on));
    lock.disabled = state.busy || Boolean(c && c.private) || noLocal;
    lock.title = noLocal ? "This hub has no local model, so its chats cannot be private"
      : obfuscating ? (on ? "Private chat: cloud models see it only obfuscated"
        : "Make this chat private: hide names and details from cloud models")
        : (on ? "Private chat: only local models see it" : "Make this chat private: local models only");
    $("fine").textContent = !on ? FINE : obfuscating
      ? "Private chat: names and other details are hidden from cloud models."
      : "Private chat: only local models on this machine see it.";
    if (!info) return;
    const model = (on && !obfuscating && info.private_ceo_model) || info.ceo_model;
    $("model-sub").textContent = `CEO ${model.replace(/^ollama(_chat)?\//, "")} · ` +
      (!on ? `${usd(info.budget_usd)} per chat` : obfuscating ? "private, obfuscated" : "private, local only");
  }

  function togglePrivate() {
    const c = chat();
    if (state.busy || (c && c.private)) return;
    if (c) { c.private = true; save(); renderChats(); }  // from the next message on, for good
    else state.draftPrivate = !state.draftPrivate;
    renderPrivacy();
  }

  // ---- plan mode ----
  // With Plan on, the CEO only reads and proposes a plan (each hire, its model and what it reserves); nothing is hired
  // until the user approves. Approving returns the chat to the hub's default mode, as the hub does.
  const YES = new Set(["approve", "approved", "yes", "y", "ok", "okay", "allow", "go ahead"]);
  const approves = (text) => YES.has(text.toLowerCase().replace(/[^\p{L}\p{N}_\s]/gu, " ").split(/\s+/).filter(Boolean).join(" "));
  const defaultMode = () => (state.info && state.info.default_mode) || "act";
  const modeOf = () => { const c = chat(); return c ? c.mode || defaultMode() : state.draftMode; };

  function renderMode() {
    const on = modeOf() === "plan";
    const button = $("plan");
    button.setAttribute("aria-pressed", String(on));
    button.disabled = state.busy;
    button.title = on ? "Plan mode: the CEO only reads and proposes a plan; nothing is hired until you approve it"
      : "Plan first: the CEO proposes a plan and waits for your approval";
  }

  function toggleMode() {
    if (state.busy) return;
    const next = modeOf() === "plan" ? "act" : "plan";
    const c = chat();
    if (c) { c.mode = next; save(); } else state.draftMode = next;
    setComposer();
  }

  function openChat(id) {
    if (state.busy) return;
    state.current = id;
    renderChats();
    renderThread();
    renderPrivacy();
    renderMode();
    $("events").replaceChildren();
    refreshAgents();
    $("input").focus();
  }

  function deleteChat(id) {
    const c = state.chats.find((x) => x.id === id);
    if (!c || (state.busy && id === state.current)) return;
    if (c.session) api(`/v1/hashiru/sessions/${encodeURIComponent(c.session)}`, { method: "DELETE" }).catch(() => {});
    state.chats = state.chats.filter((x) => x.id !== id);
    if (state.current === id) state.current = null;
    save();
    renderChats();
    renderThread();
    refreshAgents();
  }

  // ---- thread ----
  function markdown(text) {
    try {
      return window.renderMarkdown(text);
    } catch (err) {  // never let one reply break the chat: show it as plain text
      const p = el("p", "plain");
      p.textContent = text;
      return p;
    }
  }

  function renderThread() {
    const c = chat();
    const log = $("log");
    log.replaceChildren();
    const empty = !(c && c.messages.length);
    $("empty").hidden = !empty;
    if (c) c.messages.forEach((m) => log.append(messageView(m, false)));
    if (empty) $("thread").scrollTop = 0;  // the greeting starts at the top, even when the examples overflow
    else scrollDown(true);
  }

  function workKind(event) {
    if (event.kind === "tool" && event.failed) return "err";
    if (event.kind === "tool" && event.name === "hire_agent") return "hire";
    if (event.kind === "tool" && event.name === "fire_agent") return "fire";
    if (event.kind === "msg") return "msg";
    if (event.kind === "error") return "err";
    if (event.kind === "approval") return "ask";
    return "";
  }

  // Work waits for the user's approval. The next message decides, so the buttons only send one.
  const awaitsAnswer = (m) => Boolean(m && m.meta && m.meta.approval && !m.meta.answered);
  function approveView() {
    const row = el("div", "approve");
    const yes = el("button", "btn primary", "Approve");
    const no = el("button", "btn", "Deny");
    [yes, no].forEach((b) => { b.type = "button"; b.disabled = state.busy; });
    yes.addEventListener("click", () => send("approve"));
    no.addEventListener("click", () => send("deny"));
    row.append(yes, no, el("span", "hint", "or reply with what to do instead"));
    return row;
  }

  // A plan waits for the user: the next message approves it or says what to change.
  const awaitsPlan = (m) => Boolean(m && m.meta && m.meta.plan && !m.meta.answered);
  function planView() {
    const row = el("div", "approve");
    const yes = el("button", "btn primary", "Approve plan");
    yes.type = "button";
    yes.disabled = state.busy;
    yes.addEventListener("click", () => send("approve"));
    row.append(yes, el("span", "hint", "or reply with what to change"));
    return row;
  }

  function messageView(m, live) {
    if (m.role === "user") {
      const node = el("div", "msg you");
      node.append(el("div", "body", m.content));
      return node;
    }
    const node = el("div", "msg ceo");
    const avatar = el("div", "avatar");
    const face = el("img");
    face.src = "/ui/logo.svg";
    face.alt = "";
    avatar.append(face);
    node.append(avatar);
    const work = m.work || [];
    if (work.length || live) {
      const details = el("details", `work${live ? " live" : ""}`);
      details.open = m.workOpen === undefined ? Boolean(live) : m.workOpen;  // open while working, unless toggled
      const summary = el("summary");
      summary.addEventListener("click", () => { m.workOpen = !details.open; });
      const hires = work.filter((w) => w.kind === "hire").length;
      const label = live
        ? `Working${work.length ? ` · ${work.length} step${work.length === 1 ? "" : "s"}` : "…"}`
        : `Worked for ${Math.max(1, Math.round((m.meta && m.meta.secs) || 0))}s` +
          (hires ? ` · ${hires} hire${hires === 1 ? "" : "s"}` : "") + ` · ${work.length} step${work.length === 1 ? "" : "s"}`;
      summary.append(el("span", "label", label));
      details.append(summary);
      const ol = el("ol");
      work.forEach((w) => ol.append(el("li", w.kind || "", w.line)));
      details.append(ol);
      node.append(details);
    }
    const body = el("div", `md${live ? " cursor" : ""}`);
    if (m.content) body.append(markdown(m.content));
    node.append(body);
    const meta = m.meta || {};
    if (meta.error) node.append(el("div", "error", meta.error));
    else if (!live) {
      const bits = [];
      if (meta.stopped) bits.push("Stopped");
      if (meta.cost !== undefined) bits.push(`${usd(meta.cost)} this turn`);
      if (bits.length) node.append(el("div", "meta", bits.join(" · ")));
      if (awaitsAnswer(m)) node.append(approveView());
      else if (awaitsPlan(m)) node.append(planView());
    }
    return node;
  }

  function nearBottom() {
    const t = $("thread");
    return t.scrollHeight - t.scrollTop - t.clientHeight < 120;
  }
  function scrollDown(force) {
    const t = $("thread");
    if (force || nearBottom()) t.scrollTop = t.scrollHeight;
  }

  // ---- sending ----
  function history(c) {
    const out = [];
    c.messages.forEach((m) => {
      if (m.role === "user") out.push({ role: "user", content: m.content });
      else if (m.content && !(m.meta && (m.meta.error || m.meta.stopped))) out.push({ role: "assistant", content: m.content });
    });
    return out;
  }

  async function readStream(res, onChunk) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      for (let cut; (cut = buffer.indexOf("\n\n")) >= 0; ) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          onChunk(JSON.parse(data));
        }
      }
    }
  }

  async function send(text) {
    text = text.trim();
    if (state.busy || !text) return;
    let c = chat();
    if (!c) {
      c = { id: `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`, title: text.slice(0, 60),
            session: null, messages: [], updated: Date.now(), private: state.draftPrivate, mode: state.draftMode };
      state.chats.unshift(c);
      state.current = c.id;
    }
    const reply = { role: "assistant", content: "", work: [], meta: {} };
    const sent = history(c).concat([{ role: "user", content: text }]);
    const asked = c.messages[c.messages.length - 1];
    if (awaitsAnswer(asked)) asked.meta.answered = true;  // this message answers it
    if (awaitsPlan(asked)) {
      asked.meta.answered = true;
      if (approves(text)) c.mode = defaultMode();  // the plan is carried out: plan mode is over
    }
    c.messages.push({ role: "user", content: text }, reply);
    c.updated = Date.now();
    save();
    renderChats();
    renderThread();
    $("events").replaceChildren();
    let live = $("log").lastElementChild;
    let frame = 0;
    const redraw = () => {
      frame = 0;
      const next = messageView(reply, true);
      live.replaceWith(next);
      live = next;
      scrollDown(false);
    };
    const later = () => { if (!frame) frame = requestAnimationFrame(redraw); };
    state.busy = true;
    state.controller = new AbortController();
    setComposer();
    const started = performance.now();
    try {
      const sessionHeader = c.session ? { "X-Hashiru-Session": c.session } : {};
      const privateHeader = c.private ? { "X-Hashiru-Private": "1" } : {};
      const modeHeader = { "X-Hashiru-Mode": c.mode || defaultMode() };
      const res = await api("/v1/chat/completions", {
        method: "POST", signal: state.controller.signal,
        headers: Object.assign({ "Content-Type": "application/json" }, sessionHeader, privateHeader, modeHeader),
        body: JSON.stringify({ model: "hashiru", stream: true, messages: sent }),
      });
      c.session = res.headers.get("x-hashiru-session") || c.session;
      refreshAgents();
      await readStream(res, (chunk) => {
        if (chunk.error) throw new Error(chunk.error.message);
        const extra = chunk.hashiru || {};
        const delta = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) || {};
        if (extra.event) {
          addEvent(extra.event);
          if (extra.event.line) reply.work.push({ line: extra.event.line, kind: workKind(extra.event) });
          if (extra.event.kind === "tool" || extra.event.kind === "llm") scheduleAgents();
        }
        if (delta.content) reply.content += delta.content;
        if (extra.cost_usd !== undefined) reply.meta.cost = extra.cost_usd;
        if (extra.approval) reply.meta.approval = extra.approval;
        if (extra.plan) reply.meta.plan = extra.plan;
        later();
      });
    } catch (err) {
      if (err.name === "AbortError") {
        reply.meta.stopped = true;
        c.session = null;  // the hub closed the conversation; the next message starts a new one
      } else {
        reply.meta.error = err.message;
        if (err.status === 401) openSettings();
      }
    } finally {
      if (frame) cancelAnimationFrame(frame);
      reply.meta.secs = (performance.now() - started) / 1000;
      state.busy = false;
      state.controller = null;
      c.updated = Date.now();
      save();
      const done = messageView(reply, false);
      live.replaceWith(done);
      scrollDown(false);
      setComposer();
      refreshAgents();
    }
  }

  function setComposer() {
    const input = $("input");
    const c = chat();
    const last = c && c.messages[c.messages.length - 1];
    input.placeholder = awaitsAnswer(last) ? "Approve, or say what to do instead"
      : awaitsPlan(last) ? "Approve the plan, or say what to change"
        : modeOf() === "plan" ? "Describe the task: the CEO plans before it hires" : "Message HASHIRU";
    renderPrivacy();
    renderMode();
    $("send").hidden = state.busy;
    $("stop").hidden = !state.busy;
    $("send").disabled = !input.value.trim();
  }

  function grow() {
    const input = $("input");
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.4)}px`;
  }

  // ---- agents panel ----
  function addEvent(event) {
    if (!event.line) return;
    const li = el("li");
    li.append(el("span", "seq", String(event.seq)));
    const text = el("span", "", event.line);
    (event.labels || []).forEach((label) => text.append(el("span", `lab ${label}`, label)));
    li.append(text);
    $("events").append(li);
  }

  function scheduleAgents() {
    if (state.agentsTimer) return;
    state.agentsTimer = setTimeout(() => { state.agentsTimer = 0; refreshAgents(); }, 400);
  }

  async function refreshAgents() {
    const c = chat();
    const tree = $("tree");
    const totals = $("totals");
    const count = $("agent-count");
    if (!c || !c.session) {
      tree.replaceChildren(el("li", "note", c ? "This chat's agents were released. Your next message starts a new team." : "Start a chat to see the CEO and the agents it hires."));
      totals.replaceChildren();
      count.hidden = true;
      return;
    }
    let data;
    try {
      data = await (await api(`/v1/hashiru/sessions/${encodeURIComponent(c.session)}`)).json();
    } catch (err) {
      if (err.status === 404) { c.session = null; save(); return refreshAgents(); }
      return;
    }
    if (chat() !== c) return;
    if (data.private && !c.private) { c.private = true; save(); renderChats(); renderPrivacy(); }  // e.g. by default
    const live = data.agents.filter((a) => a.live && a.name !== "ceo").length;
    count.textContent = String(live);
    count.hidden = !live;
    totals.replaceChildren(...[["Budget", usd(data.budget_usd)], ["Spent", usd(data.spent_usd)],
      ["Wallet", `${usd(data.wallet.max_usd - data.wallet.spent_usd)}`]].map(([k, v]) => {
      const box = el("div");
      box.append(el("dt", "", k), el("dd", "", v));
      return box;
    }));
    const kids = {};
    data.agents.forEach((a) => { (kids[a.parent || ""] = kids[a.parent || ""] || []).push(a); });
    const branch = (parent) => {
      const ol = el("ol");
      (kids[parent] || []).forEach((a) => ol.append(agentView(a, branch)));
      return ol;
    };
    tree.replaceChildren(...branch("").children);
  }

  function agentView(a, branch) {
    const li = el("li");
    const card = el("div", `agent${a.live ? "" : " fired"}`);
    const row = el("div", "row");
    row.append(el("span", "name", a.name), el("span", "chip role", a.name === "ceo" ? "ceo" : a.role),
      el("span", `chip ${a.tier}`, a.tier));
    card.append(row, el("div", "modelname", a.model));
    const bar = el("div", "bar");
    const fill = el("i");
    const share = a.reserved_usd > 0 ? Math.max(0, Math.min(1, a.left_usd / a.reserved_usd)) : 0;
    fill.style.width = `${(share * 100).toFixed(1)}%`;
    bar.append(fill);
    const nums = el("div", "nums");
    nums.append(el("span", "", `${usd(a.left_usd)} left of ${usd(a.reserved_usd)}`),
      el("span", "", a.gpu_mb ? `${a.gpu_mb} MB` : `spent ${usd(a.spent_usd)}`));
    card.append(bar, nums);
    li.append(card);
    const below = branch(a.name);
    if (below.children.length) li.append(below);
    return li;
  }

  function toggleAgents(open) {
    const panel = $("agents");
    panel.hidden = !open;
    $("app").classList.toggle("with-agents", open);
    $("toggle-agents").setAttribute("aria-expanded", String(open));
    store.set("hashiru.agents", open);
    if (open) refreshAgents();
  }

  // ---- settings, sidebar drawer ----
  function openSettings() {
    $("api-key").value = state.key;
    if (!$("settings").open) $("settings").showModal();
  }
  function openSide() { $("app").classList.add("side-open"); $("scrim").hidden = false; }
  function closeSide() { $("app").classList.remove("side-open"); $("scrim").hidden = true; }

  async function loadInfo() {
    try {
      state.info = await (await api("/v1/hashiru/info")).json();
      if (!chat() && state.info.private_by_default) state.draftPrivate = true;  // never undo a click made before
      if (!chat() && state.info.default_mode === "plan") state.draftMode = "plan";
      setComposer();
    } catch (err) {
      if (err.status === 401) { $("model-sub").textContent = "API key needed"; openSettings(); }
    }
  }

  // ---- wiring ----
  function init() {
    applyTheme();
    EXAMPLES.forEach(([title, text]) => {
      const b = el("button", "example");
      b.type = "button";
      b.append(el("b", "", title), el("span", "", text));
      b.addEventListener("click", () => send(text));
      $("examples").append(b);
    });
    const input = $("input");
    input.addEventListener("input", () => { grow(); setComposer(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
    });
    $("composer").addEventListener("submit", (e) => {
      e.preventDefault();
      const text = input.value;
      if (!text.trim() || state.busy) return;
      input.value = "";
      grow();
      send(text);
    });
    $("stop").addEventListener("click", () => state.controller && state.controller.abort());
    $("new-chat").addEventListener("click", () => {
      if (state.busy) return;
      state.current = null;
      state.draftPrivate = Boolean(state.info && state.info.private_by_default);
      state.draftMode = defaultMode();
      renderChats(); renderThread(); setComposer(); refreshAgents(); closeSide(); input.focus();
    });
    $("private").addEventListener("click", togglePrivate);
    $("plan").addEventListener("click", toggleMode);
    $("toggle-agents").addEventListener("click", () => toggleAgents($("agents").hidden));
    $("close-agents").addEventListener("click", () => toggleAgents(false));
    $("open-side").addEventListener("click", openSide);
    $("close-side").addEventListener("click", closeSide);
    $("scrim").addEventListener("click", closeSide);
    $("open-settings").addEventListener("click", openSettings);
    $("settings").addEventListener("close", () => {
      if ($("settings").returnValue !== "save") return;
      state.key = $("api-key").value.trim();
      store.set("hashiru.key", state.key);
      loadInfo();
      refreshAgents();
    });
    $("theme").addEventListener("click", () => {
      state.theme = { auto: "light", light: "dark", dark: "auto" }[state.theme] || "auto";
      store.set("hashiru.theme", state.theme);
      applyTheme();
    });
    renderChats();
    renderThread();
    setComposer();
    if (store.get("hashiru.agents", false) && window.innerWidth > 1100) toggleAgents(true);
    else refreshAgents();
    loadInfo();
    input.focus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
