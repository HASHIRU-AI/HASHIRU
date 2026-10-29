"use strict";
// A small Markdown renderer for agents' answers. It builds DOM nodes and never parses HTML, so whatever a model
// writes stays text; links are kept only for http(s) and mailto URLs. Model output is untrusted, so nesting is
// capped (no stack overflow) and very long lines or texts are shown plain (the inline regex is not linear).
(function () {
  const MAX_BLOCK_DEPTH = 8;
  const MAX_INLINE_DEPTH = 6;
  const MAX_INLINE_LINE = 4000;
  const MAX_TEXT = 200000;
  const FENCE = /^\s*(```+|~~~+)\s*([\w+#.-]*)\s*$/;
  const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
  const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const QUOTE = /^\s*>\s?/;
  const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
  const INLINE = new RegExp([
    "(`+)([^`][\\s\\S]*?)\\1(?!`)",                                  // 1,2 code
    "\\*\\*(?=\\S)([\\s\\S]*?\\S)\\*\\*",                            // 3 bold
    "__(?=\\S)([\\s\\S]*?\\S)__",                                     // 4 bold
    "(?<![\\w*])\\*(?=[^\\s*])([\\s\\S]*?[^\\s*])\\*(?![\\w*])",     // 5 italic
    "(?<![\\w_])_(?=[^\\s_])([\\s\\S]*?[^\\s_])_(?![\\w_])",         // 6 italic
    "\\[([^\\]\\n]+)\\]\\(\\s*<?([^)\\s>]+)>?(?:\\s+\"[^\"]*\")?\\s*\\)", // 7,8 link
    "(https?:\\/\\/[^\\s<>()]+[^\\s<>().,;:!?'\"])",                   // 9 bare URL
  ].join("|"), "g");

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function plain(text) {
    const p = el("p", "plain");
    p.textContent = text;
    return p;
  }

  function link(label, href, parent, depth) {
    if (!/^(https?:|mailto:)/i.test(href)) {
      parent.append(label);
      return;
    }
    const a = el("a");
    a.href = href;
    a.rel = "noopener noreferrer nofollow";
    a.target = "_blank";
    inline(label, a, depth + 1);
    parent.append(a);
  }

  function inline(text, parent, depth) {
    const lines = text.split("\n");
    lines.forEach((line, n) => {
      if (n) parent.append(el("br"));
      if (depth > MAX_INLINE_DEPTH || line.length > MAX_INLINE_LINE) {
        parent.append(line);
        return;
      }
      let last = 0;
      INLINE.lastIndex = 0;
      for (let m; (m = INLINE.exec(line)); ) {
        if (m.index > last) parent.append(line.slice(last, m.index));
        const end = INLINE.lastIndex;
        if (m[1]) parent.append(el("code", null, m[2]));
        else if (m[3] || m[4]) { const b = el("strong"); inline(m[3] || m[4], b, depth + 1); parent.append(b); }
        else if (m[5] || m[6]) { const i = el("em"); inline(m[5] || m[6], i, depth + 1); parent.append(i); }
        else if (m[7]) link(m[7], m[8], parent, depth);
        else link(m[9], m[9], parent, depth);
        INLINE.lastIndex = end;  // nested inline() calls reuse the regex
        last = end;
      }
      if (last < line.length) parent.append(line.slice(last));
    });
  }

  const indent = (line) => line.match(/^\s*/)[0].replace(/\t/g, "    ").length;
  const blank = (line) => /^\s*$/.test(line);
  const startsBlock = (line, next) => FENCE.test(line) || HEADING.test(line) || RULE.test(line) ||
    QUOTE.test(line) || ITEM.test(line) || (line.includes("|") && next !== undefined && TABLE_SEP.test(next));

  function cells(row) {
    return row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  }

  function list(lines, i, out, depth) {
    const first = lines[i].match(ITEM);
    const base = indent(lines[i]);
    const ordered = /\d/.test(first[2]);
    const node = el(ordered ? "ol" : "ul");
    if (ordered && parseInt(first[2], 10) !== 1) node.start = parseInt(first[2], 10);
    while (i < lines.length) {
      const m = lines[i].match(ITEM);
      if (!m || indent(lines[i]) !== base || /\d/.test(m[2]) !== ordered) break;
      const body = [m[3]];
      const width = m[1].length + m[2].length + 1;
      i++;
      while (i < lines.length) {
        const next = lines[i];
        if (blank(next)) {
          if (i + 1 < lines.length && !blank(lines[i + 1]) && indent(lines[i + 1]) > base) { body.push(""); i++; continue; }
          break;
        }
        const nested = next.match(ITEM);
        if (nested && indent(next) <= base) break;
        if (!nested && indent(next) <= base && startsBlock(next, lines[i + 1])) break;
        body.push(next.slice(Math.min(indent(next), width)));
        i++;
      }
      const li = el("li");
      if (body.length === 1) inline(body[0], li, 0);
      else li.append(render(body.join("\n"), depth + 1));
      node.append(li);
    }
    out.append(node);
    return i;
  }

  function table(lines, i, out) {
    const head = cells(lines[i]);
    const node = el("table");
    const tr = el("tr");
    head.forEach((c) => { const th = el("th"); inline(c, th, 0); tr.append(th); });
    const thead = el("thead");
    thead.append(tr);
    node.append(thead);
    const body = el("tbody");
    i += 2;
    for (; i < lines.length && lines[i].includes("|") && !blank(lines[i]); i++) {
      const row = el("tr");
      cells(lines[i]).forEach((c) => { const td = el("td"); inline(c, td, 0); row.append(td); });
      body.append(row);
    }
    node.append(body);
    out.append(node);
    return i;
  }

  function render(source, depth) {
    depth = depth || 0;
    const out = document.createDocumentFragment();
    if (depth > MAX_BLOCK_DEPTH || String(source).length > MAX_TEXT) {
      out.append(plain(String(source)));
      return out;
    }
    const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      let m;
      if (blank(line)) { i++; continue; }
      if ((m = line.match(FENCE))) {
        const code = [];
        for (i++; i < lines.length && !lines[i].trim().startsWith(m[1]); i++) code.push(lines[i]);
        i++;
        const pre = el("pre");
        if (m[2]) pre.append(el("span", "lang", m[2]));
        pre.append(el("code", null, code.join("\n")));
        out.append(pre);
      } else if ((m = line.match(HEADING))) {
        const h = el(`h${Math.min(m[1].length + 1, 4)}`);
        inline(m[2], h, 0);
        out.append(h);
        i++;
      } else if (RULE.test(line)) {
        out.append(el("hr"));
        i++;
      } else if (QUOTE.test(line)) {
        const quoted = [];
        for (; i < lines.length && QUOTE.test(lines[i]); i++) quoted.push(lines[i].replace(QUOTE, ""));
        const q = el("blockquote");
        q.append(render(quoted.join("\n"), depth + 1));
        out.append(q);
      } else if (ITEM.test(line)) {
        i = list(lines, i, out, depth);
      } else if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
        i = table(lines, i, out);
      } else {
        const para = [line];
        for (i++; i < lines.length && !blank(lines[i]) && !startsBlock(lines[i], lines[i + 1]); i++) para.push(lines[i]);
        const p = el("p");
        inline(para.join("\n"), p, 0);
        out.append(p);
      }
    }
    return out;
  }

  window.renderMarkdown = (source) => render(source, 0);
})();
