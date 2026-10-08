const chatBox = document.getElementById("chatBox");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const ingestBtn = document.getElementById("ingestBtn");
const healthBtn = document.getElementById("healthBtn");
const ingestOutput = document.getElementById("ingestOutput");
const healthOutput = document.getElementById("healthOutput");
const loadGraphBtn = document.getElementById("loadGraphBtn");
const graphStatus = document.getElementById("graphStatus");
const graphModeBadge = document.getElementById("graphModeBadge");
const graphView = document.getElementById("graphView");
const entityDetail = document.getElementById("entityDetail");
const historyList = document.getElementById("historyList");
const newChatBtn = document.getElementById("newChatBtn");
const quickPromptButtons = document.querySelectorAll(".quick-prompt");
const graphSearchInput = document.getElementById("graphSearchInput");
const graphSearchBtn = document.getElementById("graphSearchBtn");
const graphResetBtn = document.getElementById("graphResetBtn");

const debugPanelToggle = document.getElementById("debugPanelToggle");
const systemStatusCard = document.getElementById("systemStatusCard");
const knowledgeGraphCard = document.getElementById("knowledgeGraphCard");
const entityDetailCard = document.getElementById("entityDetailCard");

const STORAGE_KEY = "legal_graphrag_history_v1";
const DEBUG_PANEL_STORAGE_KEY = "legal_graphrag_debug_panel_v1";

let historyData = [];
let currentConversationId = null;
let network = null;
let lastQuestion = "";
let lastAnswerGraph = { nodes: [], edges: [] };
let currentGraphMode = "answer";

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function saveHistory() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(historyData));
}

function loadHistory() {
  try {
    historyData = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    historyData = [];
  }

  if (!historyData.length) {
    const first = {
      id: uid(),
      title: "Phiên hiện tại",
      messages: [],
    };
    historyData.push(first);
    currentConversationId = first.id;
    saveHistory();
  } else {
    currentConversationId = historyData[0].id;
  }
}

function getCurrentConversation() {
  return historyData.find((item) => item.id === currentConversationId);
}

function renderHistory() {
  historyList.innerHTML = "";

  historyData.forEach((item, index) => {
    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.alignItems = "center";
    wrapper.style.gap = "6px";

    const btn = document.createElement("button");
    btn.className =
      "history-item" + (item.id === currentConversationId ? " active" : "");
    btn.type = "button";
    btn.textContent = item.title || "Phiên trò chuyện";
    btn.style.flex = "1";

    btn.addEventListener("click", () => {
      currentConversationId = item.id;
      renderHistory();
      renderConversation();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.innerHTML = "✕";
    deleteBtn.title = "Xoá cuộc trò chuyện";
    deleteBtn.style.width = "32px";
    deleteBtn.style.height = "32px";
    deleteBtn.style.borderRadius = "8px";
    deleteBtn.style.border = "1px solid #eee";
    deleteBtn.style.background = "#fff";
    deleteBtn.style.cursor = "pointer";
    deleteBtn.style.color = "#c62828";
    deleteBtn.style.fontWeight = "bold";

    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();

      const confirmDelete = confirm("Xoá cuộc trò chuyện này?");
      if (!confirmDelete) return;

      historyData.splice(index, 1);

      if (item.id === currentConversationId) {
        if (historyData.length) {
          currentConversationId = historyData[0].id;
        } else {
          createNewChat();
        }
      }

      saveHistory();
      renderHistory();
      renderConversation();
    });

    wrapper.appendChild(btn);
    wrapper.appendChild(deleteBtn);
    historyList.appendChild(wrapper);
  });
}

function addMessageToConversation(role, text, meta = {}) {
  const convo = getCurrentConversation();
  if (!convo) return;

  convo.messages.push({
    role,
    text,
    meta,
  });

  if (role === "user" && convo.title === "Phiên hiện tại") {
    convo.title = text.length > 38 ? text.slice(0, 38) + "..." : text;
  }

  saveHistory();
  renderHistory();
}

function renderConversation() {
  const convo = getCurrentConversation();
  chatBox.innerHTML = "";

  if (!convo || !convo.messages.length) {
    chatBox.innerHTML = `
      <div class="welcome-card">
        <div class="welcome-icon">⚖</div>
        <h3>Chào mừng đến với Legal GraphRAG</h3>
        <p>Bạn có thể hỏi về điều luật, chế tài, hành vi bị cấm, điều kiện áp dụng hoặc tra cứu các căn cứ trong văn bản đã nạp.</p>
      </div>
    `;
    return;
  }

  convo.messages.forEach((msg) => {
    addMessage(msg.role, msg.text, msg.meta, false);
  });

  requestAnimationFrame(() => {
    chatBox.scrollTop = chatBox.scrollHeight;
  });
}

function escapeHtml(str = "") {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAssistantText(text = "") {
  if (typeof marked !== "undefined") {
    try {
      marked.setOptions({
        breaks: false,
        gfm: true,
      });
      // Collapse 2+ blank lines into 1
      let cleaned = (text || "").replace(/\n{3,}/g, "\n\n").trim();
      let html = marked.parse(cleaned);
      html = html.replace(
        /\[Nguồn:([^\]]+)\]/g,
        '<span class="ai-citation">[Nguồn:$1]</span>'
      );
      return html;
    } catch (e) {}
  }
  // Fallback thủ công nếu marked chưa load
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="ai-strong">$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  return html.split("\n").map(l => `<p>${l}</p>`).join("");
}

function buildToggleSection(title, innerHtml, defaultOpen = false) {
  return `
    <details class="toggle-block" ${defaultOpen ? "open" : ""}>
      <summary>${title}</summary>
      <div class="toggle-content">
        ${innerHtml}
      </div>
    </details>
  `;
}

function setGraphModeBadge(mode) {
  if (!graphModeBadge) return;

  graphModeBadge.classList.remove("graph-mode-answer", "graph-mode-global");

  if (mode === "answer") {
    graphModeBadge.textContent = "Graph theo câu hỏi";
    graphModeBadge.classList.add("graph-mode-answer");
    return;
  }

  if (mode === "global") {
    graphModeBadge.textContent = "Knowledge Graph tổng";
    graphModeBadge.classList.add("graph-mode-global");
    return;
  }

  graphModeBadge.textContent = "Chưa có graph";
  graphModeBadge.classList.add("graph-mode-global");
}

function setDebugPanelVisible(isVisible) {
  [systemStatusCard, knowledgeGraphCard, entityDetailCard].forEach((el) => {
    if (!el) return;
    el.classList.toggle("debug-hidden", !isVisible);
  });

  if (debugPanelToggle) {
    debugPanelToggle.checked = isVisible;
  }

  localStorage.setItem(DEBUG_PANEL_STORAGE_KEY, JSON.stringify(isVisible));
}

function loadDebugPanelState() {
  try {
    const saved = JSON.parse(localStorage.getItem(DEBUG_PANEL_STORAGE_KEY) || "false");
    setDebugPanelVisible(Boolean(saved));
  } catch {
    setDebugPanelVisible(false);
  }
}

function addMessage(role, text, meta = {}, persist = true) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  // Avatar — chỉ bot có
  if (role === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = "⚖";
    wrapper.appendChild(avatar);
  }

  // Body container
  const body = document.createElement("div");
  body.className = "msg-body";

  const content = document.createElement("div");
  content.className = "message-content";

  if (role === "bot") {
    content.innerHTML = formatAssistantText(text);
  } else {
    content.textContent = text;
  }

  body.appendChild(content);

  // Copy button for bot messages
  if (role === "bot" && text && text.length > 10) {
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.innerHTML = "Copy";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(text).then(() => {
        copyBtn.innerHTML = "Copied";
        copyBtn.classList.add("copied");
        setTimeout(() => {
          copyBtn.innerHTML = "Copy";
          copyBtn.classList.remove("copied");
        }, 2000);
      });
    });
    actions.appendChild(copyBtn);
    body.appendChild(actions);
  }

  wrapper.appendChild(body);

  if (meta.sources && meta.sources.length) {
    const sources = document.createElement("div");
    sources.className = "message-sources-inline";
    const sourceLabels = meta.sources
      .map((s) => {
        const safeLabel = escapeHtml(s.label || "");
        const safeChunkId = escapeHtml(s.chunk_id || "");
        return `<span class="source-tag source-link" data-chunk-id="${safeChunkId}">${safeLabel}</span>`;
      })
      .join("");
    sources.innerHTML = sourceLabels;
    body.appendChild(sources);
  }

  if (meta.graphFacts && meta.graphFacts.length) {
    const facts = document.createElement("div");
    facts.className = "message-graph-facts";

    const factsHtml = meta.graphFacts
      .map((f) => `<div class="source-item">• ${escapeHtml(f)}</div>`)
      .join("");

    facts.innerHTML = buildToggleSection("Graph facts", factsHtml, false);
    body.appendChild(facts);
  }

  chatBox.appendChild(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });

  if (persist) {
    addMessageToConversation(role, text, meta);
  }

  wrapper.querySelectorAll(".source-link").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const chunkId = btn.dataset.chunkId;
      if (chunkId) {
        await loadChunkDetail(chunkId);
      }
    });
  });

  return wrapper;
}

function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

questionInput.addEventListener("input", () => {
  autoResizeTextarea(questionInput);
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

async function handleChatSubmit(e) {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  lastQuestion = question;
  currentGraphMode = "answer";
  setGraphModeBadge("answer");

  addMessage("user", question);
  questionInput.value = "";
  autoResizeTextarea(questionInput);

  // Tạo bubble bot với typing indicator
  const botWrapper = document.createElement("div");
  botWrapper.className = "message bot";
  const botAvatar = document.createElement("div");
  botAvatar.className = "msg-avatar";
  botAvatar.textContent = "⚖";
  botWrapper.appendChild(botAvatar);
  const botBody = document.createElement("div");
  botBody.className = "msg-body";
  const contentEl = document.createElement("div");
  contentEl.className = "message-content";
  contentEl.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  botBody.appendChild(contentEl);
  botWrapper.appendChild(botBody);
  chatBox.appendChild(botWrapper);
  botWrapper.scrollIntoView({ behavior: "smooth", block: "end" });

  let fullText = "";
  let metaChunks = [];
  let metaGraph = null;
  let hasPdfData = false;
  let sourcesRendered = false;

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let msg;
        try { msg = JSON.parse(line.slice(6)); } catch { continue; }

        if (msg.type === "meta") {
          metaChunks = msg.chunks || [];
          metaGraph = msg.answer_graph;
          hasPdfData = msg.has_pdf_data || false;
        } else if (msg.type === "token") {
          fullText += msg.text;
          contentEl.innerHTML = formatAssistantText(fullText);
          botWrapper.scrollIntoView({ behavior: "smooth", block: "end" });
        } else if (msg.type === "done") {
          hasPdfData = msg.has_pdf_data ?? hasPdfData;

          // Render sources
          if (hasPdfData && metaChunks.length && !sourcesRendered) {
            sourcesRendered = true;
            const sources = metaChunks.map((c, i) => {
              const legal = [c.dieu, c.khoan].filter(Boolean).join(" | ");
              const label = legal
                ? `[Nguồn ${i+1}] ${c.source} - trang ${c.page} | ${legal}`
                : `[Nguồn ${i+1}] ${c.source} - trang ${c.page}`;
              return { label, chunk_id: c.chunk_id || "" };
            });
            const sourcesDiv = document.createElement("div");
            sourcesDiv.className = "message-sources-inline";
            sourcesDiv.innerHTML = sources.map(s =>
              `<span class="source-tag source-link" data-chunk="${s.chunk_id}">${s.label}</span>`
            ).join("");
            botBody.appendChild(sourcesDiv);
            sourcesDiv.querySelectorAll(".source-link").forEach(btn => {
              btn.addEventListener("click", () => handleSourceClick(btn.dataset.chunk));
            });
          }

          // Render graph
          if (metaGraph && Array.isArray(metaGraph.nodes)) {
            lastAnswerGraph = metaGraph;
            renderGraph(metaGraph, "Graph theo câu hỏi");
            setGraphModeBadge("answer");
          }

          // Lưu vào history
          const conv = getCurrentConversation();
          if (conv) {
            conv.messages.push({ role: "bot", text: fullText, sources: [], graphFacts: [] });
            if (conv.messages.length === 2) conv.title = question.slice(0, 40);
            saveHistory();
            renderHistory();
          }
        } else if (msg.type === "error") {
          contentEl.innerHTML = "Có lỗi: " + (msg.message || "");
        }
      }
    }
  } catch (err) {
    contentEl.innerHTML = "Có lỗi khi kết nối: " + err.message;
  }
}

async function handleIngest() {
  ingestOutput.textContent = "Đang nạp PDF...";
  try {
    const res = await fetch("/api/ingest", { method: "POST" });
    const data = await res.json();
    ingestOutput.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    ingestOutput.textContent = "Lỗi: " + err.message;
  }
}

async function handleHealth() {
  healthOutput.textContent = "Đang kiểm tra...";
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    healthOutput.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    healthOutput.textContent = "Lỗi: " + err.message;
  }
}

function buildGraphOptions() {
  return {
    autoResize: true,
    layout: {
      improvedLayout: true,
      randomSeed: 12,
    },
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -60,
        centralGravity: 0.015,
        springLength: 160,
        springConstant: 0.05,
        damping: 0.6,
        avoidOverlap: 1,
      },
      stabilization: {
        enabled: true,
        iterations: 180,
        updateInterval: 20,
        fit: true,
      },
      minVelocity: 0.75,
      timestep: 0.5,
    },
    interaction: {
      hover: true,
      navigationButtons: true,
      keyboard: true,
      tooltipDelay: 120,
      zoomView: true,
      dragView: true,
      dragNodes: true,
    },
    nodes: {
      shape: "dot",
      scaling: {
        min: 12,
        max: 26,
      },
      borderWidth: 2,
      shadow: {
        enabled: true,
        color: "rgba(198,40,40,0.18)",
        size: 12,
        x: 0,
        y: 4,
      },
      font: {
        color: "#111827",
        size: 15,
        face: "Inter",
        bold: true,
        strokeWidth: 4,
        strokeColor: "#ffffff",
      },
    },
    edges: {
      arrows: {
        to: {
          enabled: true,
          scaleFactor: 0.7,
        },
      },
      width: 1.8,
      selectionWidth: 2.5,
      smooth: {
        enabled: true,
        type: "continuous",
        roundness: 0.18,
      },
      color: {
        inherit: false,
        opacity: 0.82,
      },
      font: {
        align: "middle",
        size: 10,
        color: "#4b1111",
        background: "rgba(255,255,255,0.9)",
        strokeWidth: 2,
        strokeColor: "#ffffff",
      },
    },
  };
}

function styleNodes(rawNodes, rawEdges) {
  const degreeMap = {};

  rawEdges.forEach((e) => {
    degreeMap[e.from] = (degreeMap[e.from] || 0) + 1;
    degreeMap[e.to] = (degreeMap[e.to] || 0) + 1;
  });

  return rawNodes.map((n) => {
    const degree = degreeMap[n.id] || 1;
    const size = Math.min(14 + degree * 1.6, 30);
    const isSeed = n.group === "seed";

    return {
      ...n,
      value: degree,
      size: isSeed ? Math.min(size + 4, 34) : size,
      shape: "dot",
      color: {
        background: isSeed ? "#8e1d1d" : (degree >= 6 ? "#b71c1c" : "#d32f2f"),
        border: "#8e1d1d",
        highlight: {
          background: "#ef5350",
          border: "#b71c1c",
        },
        hover: {
          background: "#e53935",
          border: "#8e1d1d",
        },
      },
      font: {
        color: "#111827",
        size: isSeed ? 16 : (degree >= 6 ? 16 : 14),
        face: "Inter",
        bold: true,
        strokeWidth: 4,
        strokeColor: "#ffffff",
      },
      title: `${n.label}\nBậc kết nối: ${degree}`,
    };
  });
}

function styleEdges(rawEdges) {
  return rawEdges.map((e) => ({
    ...e,
    arrows: "to",
    color: {
      color: e.label === "LIEN_QUAN" ? "#e9b3b3" : "#c62828",
      highlight: "#8e1d1d",
      hover: "#b71c1c",
    },
    font: {
      color: "#4b1111",
      size: 10,
      strokeWidth: 2,
      strokeColor: "#ffffff",
      background: "rgba(255,255,255,0.9)",
    },
    smooth: {
      enabled: true,
      type: "continuous",
      roundness: 0.16,
    },
  }));
}

function highlightGraphByQuestion(question) {
  if (!network || !question) return;

  const terms = question
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 2);

  const allNodes = network.body.data.nodes.get();

  const matched = allNodes
    .filter((n) => terms.some((t) => n.label.toLowerCase().includes(t)))
    .map((n) => n.id);

  if (!matched.length) return;

  network.selectNodes(matched);
  network.focus(matched[0], {
    scale: 1.3,
    animation: {
      duration: 400,
      easingFunction: "easeInOutQuad",
    },
  });
}

function styleNodes(nodes, edges) {
  const edgeCounts = {};
  edges.forEach(e => {
    edgeCounts[e.from] = (edgeCounts[e.from] || 0) + 1;
    edgeCounts[e.to]   = (edgeCounts[e.to]   || 0) + 1;
  });
  return nodes.map(n => {
    const isSeed   = n.group === "seed";
    const degree   = edgeCounts[n.id] || 0;
    const baseSize = isSeed ? 22 : Math.min(10 + degree * 2, 20);
    return {
      id: n.id,
      label: n.label,
      shape: "dot",
      size: baseSize,
      color: {
        background: isSeed ? "#c62828" : "#ffffff",
        border:     isSeed ? "#8e1d1d" : "#c62828",
        highlight:  { background: "#ef5350", border: "#c62828" },
        hover:      { background: "#ffebee", border: "#c62828" },
      },
      font: {
        color: isSeed ? "#ffffff" : "#1b1f27",
        size: 12,
        face: "Inter, system-ui, sans-serif",
      },
    };
  });
}

function styleEdges(edges) {
  return edges.map(e => ({
    from:  e.from,
    to:    e.to,
    label: e.label || "",
    color: { color: "rgba(198,40,40,0.25)", highlight: "#c62828", hover: "#c62828" },
    font:  { size: 10, color: "#7b8190", align: "middle" },
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    smooth: { type: "dynamic" },
    width: 1,
  }));
}

function buildGraphOptions() {
  return {
    physics: {
      enabled: true,
      stabilization: { iterations: 150, fit: true },
      barnesHut: { gravitationalConstant: -5000, springLength: 100, damping: 0.3 },
    },
    interaction: { hover: true, tooltipDelay: 200, hideEdgesOnDrag: true },
    nodes: { borderWidth: 1.5 },
    edges: { selectionWidth: 2 },
    layout: { improvedLayout: true },
  };
}

function renderGraph(graphData, modeLabel = "Knowledge Graph") {
  const rawNodes = Array.isArray(graphData?.nodes) ? graphData.nodes : [];
  const rawEdges = Array.isArray(graphData?.edges) ? graphData.edges : [];

  if (!rawNodes.length) {
    graphView.innerHTML = `<div class="graph-placeholder">Không có graph phù hợp để hiển thị</div>`;
    graphStatus.textContent = `${modeLabel}: không có dữ liệu.`;
    network = null;
    return;
  }

  const nodes = new vis.DataSet(styleNodes(rawNodes, rawEdges));
  const edges = new vis.DataSet(styleEdges(rawEdges));

  graphView.innerHTML = "";

  network = new vis.Network(graphView, { nodes, edges }, buildGraphOptions());

  network.once("stabilizationIterationsDone", () => {
    network.setOptions({ physics: { enabled: false } });
    network.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" } });
  });

  network.on("click", async params => {
    if (!params.nodes.length) return;
    await loadEntityDetail(params.nodes[0]);
  });

  graphStatus.textContent = `${modeLabel}: ${rawNodes.length} node / ${rawEdges.length} cạnh.`;
}

function searchGraphNode() {
  if (!network || !graphSearchInput) return;
  const keyword = graphSearchInput.value.trim().toLowerCase();
  if (!keyword) return;

  const allNodes = network.body.data.nodes.get();
  const matched  = allNodes.find(n => (n.label || "").toLowerCase().includes(keyword));
  if (!matched) {
    graphStatus.textContent = `Không tìm thấy entity: ${keyword}`;
    return;
  }
  network.selectNodes([matched.id]);
  network.focus(matched.id, { scale: 1.4, animation: { duration: 400, easingFunction: "easeInOutQuad" } });
  graphStatus.textContent = `Đã focus: ${matched.label}`;
  loadEntityDetail(matched.id);
}

function resetGraphView() {
  if (currentGraphMode === "answer" && lastAnswerGraph.nodes?.length) {
    renderGraph(lastAnswerGraph, "Graph theo câu hỏi");
    setGraphModeBadge("answer");
    return;
  }
  if (!network) return;
  network.unselectAll();
  network.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" } });
  graphStatus.textContent = "Đã reset graph.";
}

async function loadGraph() {
  currentGraphMode = "global";
  setGraphModeBadge("global");
  graphStatus.textContent = "Đang tải Knowledge Graph tổng...";

  try {
    const res = await fetch("/api/graph?limit=140");
    const data = await res.json();

    if (data.status !== "ok") {
      graphStatus.textContent = "Lỗi tải graph: " + (data.message || "Unknown error");
      return;
    }

    renderGraph(data.graph, "Knowledge Graph tổng");
  } catch (err) {
    graphStatus.textContent = "Lỗi tải graph: " + err.message;
  }
}

async function loadEntityDetail(entityName) {
  entityDetail.innerHTML = `Đang tải chi tiết cho: <strong>${escapeHtml(entityName)}</strong>...`;

  try {
    const res = await fetch(`/api/entity?name=${encodeURIComponent(entityName)}`);
    const data = await res.json();

    if (data.status !== "ok") {
      entityDetail.textContent = "Không tải được chi tiết entity.";
      return;
    }

    const entity = data.data.entity;
    const chunks = data.data.chunks || [];
    const relations = data.data.relations || [];

    entityDetail.innerHTML = `
      <div class="entity-title">${escapeHtml(entity)}</div>

      <div class="entity-section">
        <div class="entity-section-title">Quan hệ liên quan</div>
        ${
          relations.length
            ? relations
                .map(
                  (r) =>
                    `<div class="entity-relation">• ${escapeHtml(r.source)} --[${escapeHtml(r.relation)}]--> ${escapeHtml(r.target)}</div>`
                )
                .join("")
            : '<div class="entity-empty">Không có quan hệ nổi bật.</div>'
        }
      </div>

      <div class="entity-section">
        <div class="entity-section-title">Đoạn luật liên quan</div>
        ${
          chunks.length
            ? chunks
                .map(
                  (c) => `
                    <div class="entity-chunk">
                      <div class="entity-chunk-meta">${escapeHtml(c.source)} - trang ${escapeHtml(String(c.page))}</div>
                      <div class="entity-chunk-text">${escapeHtml(c.content)}</div>
                    </div>
                  `
                )
                .join("")
            : '<div class="entity-empty">Không tìm thấy chunk liên quan.</div>'
        }
      </div>
    `;
  } catch (err) {
    entityDetail.textContent = "Lỗi tải chi tiết entity: " + err.message;
  }
}

async function loadChunkDetail(chunkId) {
  entityDetail.innerHTML = `Đang tải đoạn luật nguồn...`;

  try {
    const res = await fetch(`/api/chunk?id=${encodeURIComponent(chunkId)}`);
    const data = await res.json();

    if (data.status !== "ok") {
      entityDetail.textContent = "Không tải được nội dung nguồn.";
      return;
    }

    const c = data.data;

    entityDetail.innerHTML = `
      <div class="entity-title">Nguồn luật</div>
      <div class="entity-section">
        <div class="entity-chunk">
          <div class="entity-chunk-meta">${escapeHtml(c.source)} - trang ${escapeHtml(String(c.page))}</div>
          ${
            c.summary
              ? `<div class="entity-section-title">Tóm tắt</div><div class="entity-relation">${escapeHtml(c.summary)}</div>`
              : ""
          }
          <div class="entity-section-title">Nội dung</div>
          <div class="entity-chunk-text">${escapeHtml(c.content)}</div>
        </div>
      </div>
    `;
  } catch (err) {
    entityDetail.textContent = "Lỗi tải nội dung nguồn: " + err.message;
  }
}

function createNewChat() {
  const convo = {
    id: uid(),
    title: "Phiên hiện tại",
    messages: [],
  };
  historyData.unshift(convo);
  currentConversationId = convo.id;
  saveHistory();
  renderHistory();
  renderConversation();
}

chatForm.addEventListener("submit", handleChatSubmit);
ingestBtn.addEventListener("click", handleIngest);
healthBtn.addEventListener("click", handleHealth);
loadGraphBtn.addEventListener("click", loadGraph);

if (graphSearchBtn) {
  graphSearchBtn.addEventListener("click", searchGraphNode);
}

if (graphSearchInput) {
  graphSearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      searchGraphNode();
    }
  });
}

if (graphResetBtn) {
  graphResetBtn.addEventListener("click", resetGraphView);
}

if (debugPanelToggle) {
  debugPanelToggle.addEventListener("change", (e) => {
    setDebugPanelVisible(Boolean(e.target.checked));
  });
}

newChatBtn.addEventListener("click", createNewChat);

quickPromptButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    questionInput.value = btn.textContent.trim();
    autoResizeTextarea(questionInput);
    questionInput.focus();
  });
});

loadHistory();
renderHistory();
renderConversation();
autoResizeTextarea(questionInput);
setGraphModeBadge("global");
loadDebugPanelState();
initUploadZone();
loadPdfList();

// =========================
// UPLOAD PDF UI
// =========================
async function loadPdfList() {
  const listEl = document.getElementById("pdfList");
  if (!listEl) return;
  try {
    const res = await fetch("/api/pdfs");
    const data = await res.json();
    if (!data.files || !data.files.length) {
      listEl.innerHTML = '<p class="muted" style="font-size:13px;margin:8px 0">Chưa có file PDF nào.</p>';
      return;
    }
    listEl.innerHTML = data.files.map(f => `
      <div class="pdf-item" id="pdfitem-${f.name}">
        <span class="pdf-name">📄 ${f.name}</span>
        <span class="pdf-size">${f.size_kb} KB</span>
        <button class="pdf-delete-btn" onclick="deletePdf('${f.name}')" title="Xoá">✕</button>
      </div>
    `).join("");
  } catch(e) {
    listEl.innerHTML = '<p class="muted" style="font-size:13px">Lỗi tải danh sách.</p>';
  }
}

async function deletePdf(filename) {
  if (!confirm(`Xoá file "${filename}"?`)) return;
  try {
    await fetch(`/api/pdfs/${encodeURIComponent(filename)}`, { method: "DELETE" });
    loadPdfList();
  } catch(e) { alert("Lỗi xoá file: " + e.message); }
}

function initUploadZone() {
  const zone = document.getElementById("uploadZone");
  const input = document.getElementById("uploadInput");
  const status = document.getElementById("uploadStatus");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    uploadFiles(e.dataTransfer.files);
  });
  input.addEventListener("change", () => uploadFiles(input.files));
}

async function uploadFiles(files) {
  const status = document.getElementById("uploadStatus");
  if (!files || !files.length) return;
  const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith(".pdf"));
  if (!pdfs.length) { status.textContent = "Chỉ chấp nhận file PDF."; return; }

  status.textContent = `Đang tải lên ${pdfs.length} file...`;
  const form = new FormData();
  pdfs.forEach(f => form.append("files", f));

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    status.textContent = data.message || "Tải lên thành công.";
    loadPdfList();
  } catch(e) { status.textContent = "Lỗi: " + e.message; }
}
// =========================
// DARK MODE
// =========================
function initTheme() {
  const saved = localStorage.getItem("legal_rag_theme");
  if (saved) {
    document.documentElement.setAttribute("data-theme", saved);
  } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
  updateThemeIcon();
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("legal_rag_theme", next);
  updateThemeIcon();
}

function updateThemeIcon() {
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  btn.textContent = isDark ? "☀️" : "🌙";
}

// Create theme toggle button
const themeBtn = document.createElement("button");
themeBtn.id = "themeToggle";
themeBtn.className = "theme-toggle";
themeBtn.title = "Chuyển đổi sáng/tối";
themeBtn.addEventListener("click", toggleTheme);
document.body.appendChild(themeBtn);
initTheme();