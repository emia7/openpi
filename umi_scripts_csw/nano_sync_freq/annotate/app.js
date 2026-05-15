/* global document, window, URL, fetch, FileReader, Blob, URLSearchParams */
/**
 * 标定时间轴：系统参考层 + 可编辑起停、HTML5 视频；快捷键 a/c、e、Space；
 * e：选中竖线后，把该时刻从起表挪到停表或反之（只动此点，不与其他点互换）；
 * ⌘/Ctrl+Z 撤销、⇧+⌘/Ctrl+Z 重做（输入框内交给浏览器）；导出 JSON。
 */
(function () {
  "use strict";

  const v = document.getElementById("v");
  const rail = document.getElementById("rail");
  const playhead = document.getElementById("playhead");
  const timeReadout = document.getElementById("timeReadout");
  const layerSys = document.getElementById("layerSys");
  const layerUser = document.getElementById("layerUser");
  const layerPreview = document.getElementById("layerPreview");
  const layerMan = document.getElementById("layerMan");
  const onlyUser = document.getElementById("onlyUserLayer");
  const showPreview = document.getElementById("showPreview");
  const tblS = document.querySelector("#tblS tbody");
  const tblT = document.querySelector("#tblT tbody");
  const tblMan = document.querySelector("#tblMan tbody");
  const fileMarkers = document.getElementById("fileMarkers");
  const fileVideo = document.getElementById("fileVideo");
  const viewRangeEl = document.getElementById("viewRange");

  /** @type {object | null} */
  let raw = null;
  let sysS = [];
  let sysT = [];
  let userS = [];
  let userT = [];
  let durationSec = 0;
  let manualList = [];
  let previewTailSec = 0.12;

  /** 时间轴可见窗口 [viewStart, viewEnd]（秒），用于缩放/平移 */
  let viewStart = 0;
  let viewEnd = 0;
  /** @type {{ pointerId: number, x0: number, v0: number, e0: number, moved: boolean } | null} */
  let railDrag = null;
  /** 时间轴上最后一次点击的可编辑竖线，用于 <kbd>e</kbd> 起停互换 */
  let pickMarker = null;

  const MAX_UNDO = 120;
  /** @type {{ userS: number[], userT: number[] }[]} */
  let undoStack = [];
  /** @type {{ userS: number[], userT: number[] }[]} */
  let redoStack = [];

  function snapshotState() {
    return { userS: userS.slice(), userT: userT.slice() };
  }

  function applyState(s) {
    userS = s.userS.slice();
    userT = s.userT.slice();
    pickMarker = null;
    renderTableS();
    renderTableT();
    redrawTimeline();
  }

  function recordBeforeChange() {
    undoStack.push(snapshotState());
    if (undoStack.length > MAX_UNDO) {
      undoStack.shift();
    }
    redoStack = [];
  }

  function undoOnce() {
    if (undoStack.length === 0) {
      return;
    }
    redoStack.push(snapshotState());
    applyState(undoStack.pop());
  }

  function redoOnce() {
    if (redoStack.length === 0) {
      return;
    }
    undoStack.push(snapshotState());
    applyState(redoStack.pop());
  }

  const MIN_VIEW_SPAN = 0.25;
  const ZOOM_FACTOR = 1.28;
  const ZOOM_WHEEL = 1.12;
  const PAN_STEP_FRACTION = 0.35;

  function numArr(x) {
    if (!x || !Array.isArray(x)) return [];
    return x.map((n) => Number(n));
  }

  function deepCopyNumArr(a) {
    return numArr(a).slice();
  }

  function typingFocus() {
    const el = document.activeElement;
    if (!el) return false;
    const t = el.tagName;
    if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  function pairStartStopTimes(startTimes, stopTimes, durationSecIn) {
    const s = startTimes.slice().sort((a, b) => a - b);
    const t = stopTimes.slice().sort((a, b) => a - b);
    const w = [];
    const clips = [];
    let j = 0;
    const n = t.length;
    for (const ts of s) {
      while (j < n && t[j] <= ts + 1e-6) {
        w.push(
          `drop_stop_before_or_at_start: ${t[j].toFixed(3)} (start=${ts.toFixed(3)})`
        );
        j += 1;
      }
      if (j >= n) {
        w.push(`unpaired_start: ${ts.toFixed(3)}`);
        continue;
      }
      let te = t[j];
      j += 1;
      if (durationSecIn != null) {
        if (ts > durationSecIn) {
          w.push(`start_after_effective_duration: ${ts.toFixed(3)}`);
          continue;
        }
        if (te > durationSecIn + 1e-3) {
          w.push(
            `clip_end_clamped: start=${ts.toFixed(3)} stop ${te.toFixed(3)} -> ${Number(durationSecIn).toFixed(3)}`
          );
          te = durationSecIn;
        }
      }
      if (te > ts + 1e-6) {
        clips.push({ index: clips.length, t_start: ts, t_end: te });
      }
    }
    while (j < n) {
      w.push(`unpaired_stop: ${t[j].toFixed(3)}`);
      j += 1;
    }
    return { clips, warnings: w };
  }

  function applyStopBeepTail(
    pr,
    tailSec,
    durationIn,
    safetyBeforeNextStart
  ) {
    if (tailSec <= 0 || !pr.clips.length) return pr;
    const w = pr.warnings.slice();
    const out = [];
    const n = pr.clips.length;
    for (let i = 0; i < n; i += 1) {
      const c = pr.clips[i];
      const onset = c.t_end;
      let te = onset + tailSec;
      if (i + 1 < n) {
        te = Math.min(
          te,
          pr.clips[i + 1].t_start - safetyBeforeNextStart
        );
      }
      if (durationIn != null && durationIn > 0) {
        te = Math.min(te, durationIn);
      }
      if (te < onset + tailSec - 1e-4) {
        w.push(
          `clip_${c.index}_stop_tail_clamped: 计划尾音 ${tailSec.toFixed(3)}s，实际 ${(te - onset).toFixed(3)}s`
        );
      }
      out.push({
        index: c.index,
        t_start: c.t_start,
        t_end: te,
        t_stop_onset: onset,
      });
    }
    return { clips: out, warnings: w };
  }

  function getMaxD() {
    const vd = v.duration && !Number.isNaN(v.duration) && v.duration > 0
      ? v.duration
      : 0;
    return Math.max(Number(durationSec) || 0, vd, 0.01);
  }

  function getViewSpan() {
    return Math.max(viewEnd - viewStart, 1e-6);
  }

  /** 将 t（秒）映射为可见窗口内的 0–100% */
  function timeToPctInView(t) {
    return (100 * (t - viewStart)) / getViewSpan();
  }

  function clampView() {
    const maxD = getMaxD();
    let s = viewStart;
    let e = viewEnd;
    if (e - s < MIN_VIEW_SPAN) {
      const c = (s + e) / 2;
      s = c - MIN_VIEW_SPAN / 2;
      e = c + MIN_VIEW_SPAN / 2;
    }
    if (e - s > maxD) {
      s = 0;
      e = maxD;
    }
    if (s < 0) {
      e -= s;
      s = 0;
    }
    if (e > maxD) {
      s -= e - maxD;
      e = maxD;
    }
    if (s < 0) s = 0;
    viewStart = s;
    viewEnd = e;
  }

  function fitView() {
    const maxD = getMaxD();
    viewStart = 0;
    viewEnd = maxD;
    updateViewReadout();
    redrawTimeline();
    updatePlayhead();
  }

  function zoomAt(focalTime, factor) {
    const maxD = getMaxD();
    if (maxD < MIN_VIEW_SPAN) return;
    const span = getViewSpan();
    const t = Math.max(viewStart, Math.min(viewEnd, focalTime));
    const r = span > 0 ? (t - viewStart) / span : 0.5;
    let newSpan = span * factor;
    newSpan = Math.max(MIN_VIEW_SPAN, Math.min(newSpan, maxD));
    let ns = t - r * newSpan;
    let ne = ns + newSpan;
    if (ne > maxD) {
      ne = maxD;
      ns = maxD - newSpan;
    }
    if (ns < 0) {
      ns = 0;
      ne = newSpan;
    }
    viewStart = ns;
    viewEnd = ne;
    clampView();
    updateViewReadout();
    redrawTimeline();
    updatePlayhead();
  }

  function viewCenteredOn(focalTime) {
    const maxD = getMaxD();
    const span = getViewSpan();
    if (span >= maxD - 1e-6) {
      return;
    }
    const t = Math.max(0, Math.min(focalTime, maxD));
    let ns = t - span / 2;
    let ne = t + span / 2;
    if (ns < 0) {
      ns = 0;
      ne = span;
    }
    if (ne > maxD) {
      ne = maxD;
      ns = maxD - span;
    }
    viewStart = Math.max(0, ns);
    viewEnd = Math.min(maxD, ne);
    clampView();
    updateViewReadout();
    redrawTimeline();
    updatePlayhead();
  }

  function panByTime(delta) {
    const maxD = getMaxD();
    const span = getViewSpan();
    if (span >= maxD) return;
    let ns = viewStart + delta;
    let ne = viewEnd + delta;
    if (ns < 0) {
      ne = span;
      ns = 0;
    }
    if (ne > maxD) {
      ne = maxD;
      ns = maxD - span;
    }
    viewStart = ns;
    viewEnd = ne;
    updateViewReadout();
    redrawTimeline();
    updatePlayhead();
  }

  function timeUnderPointer(clientX, el) {
    const r = el.getBoundingClientRect();
    const x = clientX - r.left;
    const ratio = r.width > 0 ? x / r.width : 0;
    return viewStart + ratio * getViewSpan();
  }

  function updateViewReadout() {
    if (!viewRangeEl) return;
    const maxD = getMaxD();
    const w = getViewSpan();
    if (maxD <= 0) {
      viewRangeEl.textContent = "窗口: —";
      return;
    }
    if (w >= maxD - 1e-3) {
      viewRangeEl.textContent =
        "可见: 全长 0.0s – " + maxD.toFixed(1) + " s";
    } else {
      const scale = (maxD / w).toFixed(1);
      viewRangeEl.textContent =
        "可见: " +
        viewStart.toFixed(1) +
        " – " +
        viewEnd.toFixed(1) +
        " s  （窗宽 " +
        w.toFixed(1) +
        " s，约 " +
        scale +
        "× 拉伸）";
    }
  }

  function clearLayer(el) {
    el.textContent = "";
  }

  function addMarkers(layer, times, className) {
    const labelMap = {
      systemS: "系统·起",
      systemT: "系统·停",
      userS: "可编辑·起",
      userT: "可编辑·停",
    };
    const lab = labelMap[className] || "";
    for (let idx = 0; idx < times.length; idx += 1) {
      const t = times[idx];
      if (t == null || Number.isNaN(t)) continue;
      if (t < viewStart - 1e-9 || t > viewEnd + 1e-9) continue;
      const d = document.createElement("div");
      d.className = "mk " + className;
      if (className === "userS" || className === "userT") {
        d.dataset.k = className === "userS" ? "s" : "t";
        d.dataset.idx = String(idx);
        if (
          pickMarker &&
          pickMarker.k === d.dataset.k &&
          pickMarker.idx === idx
        ) {
          d.classList.add("mk--pick");
        }
        d.title =
          (lab ? lab + " " : "") +
          t.toFixed(3) +
          "s — 选中后按 e：此起停互换类别（该时刻移到另一张表）";
      } else {
        d.title = (lab ? lab + " " : "") + t.toFixed(3) + "s";
      }
      d.style.left = timeToPctInView(t) + "%";
      layer.appendChild(d);
    }
  }

  /**
   * 选中某一「起」或「停」竖线后：把该时刻从起表移到停表，或从停表移到起表（仅此一点，不与别的点对换）。
   */
  function reclassifyPickAtSelection() {
    if (!pickMarker) {
      return false;
    }
    if (pickMarker.k === "s") {
      const i = pickMarker.idx;
      if (i < 0 || i >= userS.length) {
        return false;
      }
      const tVal = userS[i];
      userS.splice(i, 1);
      userT.push(tVal);
      userS.sort(sortNum);
      userT.sort(sortNum);
      const ni = userT.indexOf(tVal);
      pickMarker = ni >= 0 ? { k: "t", idx: ni } : null;
      return true;
    }
    const j = pickMarker.idx;
    if (j < 0 || j >= userT.length) {
      return false;
    }
    const tVal = userT[j];
    userT.splice(j, 1);
    userS.push(tVal);
    userS.sort(sortNum);
    userT.sort(sortNum);
    const ni = userS.indexOf(tVal);
    pickMarker = ni >= 0 ? { k: "s", idx: ni } : null;
    return true;
  }

  function drawPreviewClips(clips) {
    clearLayer(layerPreview);
    for (const c of clips) {
      const t0 = Math.max(c.t_start, viewStart);
      const t1 = Math.min(c.t_end, viewEnd);
      if (t0 >= t1 - 1e-9) continue;
      const div = document.createElement("div");
      div.className = "clipSeg";
      const left = timeToPctInView(t0);
      const w = Math.max(0, timeToPctInView(t1) - left);
      div.style.left = left + "%";
      div.style.width = w + "%";
      div.title = `#${c.index} [${c.t_start.toFixed(2)}, ${c.t_end.toFixed(2)}]`;
      div.textContent = String(c.index);
      layerPreview.appendChild(div);
    }
  }

  function drawManual() {
    clearLayer(layerMan);
    for (const m of manualList) {
      const t = m.t != null ? Number(m.t) : NaN;
      if (Number.isNaN(t)) continue;
      if (t < viewStart - 1e-9 || t > viewEnd + 1e-9) continue;
      const d = document.createElement("div");
      d.className = "mk";
      d.style.left = timeToPctInView(t) + "%";
      d.style.background = "var(--man)";
      d.title = [m.reason, m.label, m.note].filter(Boolean).join(" — ");
      layerMan.appendChild(d);
    }
  }

  function redrawTimeline() {
    clearLayer(layerSys);
    clearLayer(layerUser);
    if (!onlyUser.checked) {
      addMarkers(layerSys, sysS, "systemS");
      addMarkers(layerSys, sysT, "systemT");
    }
    addMarkers(layerUser, userS, "userS");
    addMarkers(layerUser, userT, "userT");
    if (showPreview.checked) {
      const pr0 = pairStartStopTimes(
        userS,
        userT,
        durationSec > 0 ? durationSec : null
      );
      const pr1 = applyStopBeepTail(
        pr0,
        previewTailSec,
        durationSec > 0 ? durationSec : null,
        0.01
      );
      drawPreviewClips(pr1.clips);
    } else {
      clearLayer(layerPreview);
    }
    drawManual();
    layerSys.style.display = onlyUser.checked ? "none" : "block";
    layerPreview.style.display = showPreview.checked ? "block" : "none";
  }

  function updatePlayhead() {
    const t = v.currentTime || 0;
    const d = v.duration;
    playhead.classList.remove("playhead--before", "playhead--after");
    if (d && !Number.isNaN(d) && d > 0) {
      if (t < viewStart - 1e-6) {
        playhead.style.left = "0%";
        playhead.classList.add("playhead--before");
      } else if (t > viewEnd + 1e-6) {
        playhead.style.left = "100%";
        playhead.classList.add("playhead--after");
      } else {
        playhead.style.left = timeToPctInView(t) + "%";
      }
    } else {
      playhead.style.left = "0%";
    }
    timeReadout.textContent =
      t.toFixed(3) + "s / " + (d && !Number.isNaN(d) ? d.toFixed(3) : "0.000") + "s";
  }

  function sortNum(a, b) {
    return a - b;
  }

  function renderTableS() {
    tblS.innerHTML = "";
    userS.forEach((val, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${i + 1}</td><td><input class="tcell" type="text" data-idx="${i}" data-k="s" value="${val}" /></td><td><button type="button" data-act="delS" data-i="${i}">删</button></td>`;
      tblS.appendChild(tr);
    });
  }

  function renderTableT() {
    tblT.innerHTML = "";
    userT.forEach((val, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${i + 1}</td><td><input class="tcell" type="text" data-idx="${i}" data-k="t" value="${val}" /></td><td><button type="button" data-act="delT" data-i="${i}">删</button></td>`;
      tblT.appendChild(tr);
    });
  }

  function applyTableFromInputs() {
    const insS = tblS.querySelectorAll('input[data-k="s"]');
    const insT = tblT.querySelectorAll('input[data-k="t"]');
    const ns = [];
    insS.forEach((inp) => {
      const x = parseFloat(String(inp.value).replace(/,/g, "."));
      if (!Number.isNaN(x)) ns.push(x);
    });
    const nt = [];
    insT.forEach((inp) => {
      const x = parseFloat(String(inp.value).replace(/,/g, "."));
      if (!Number.isNaN(x)) nt.push(x);
    });
    userS = ns;
    userT = nt;
  }

  function renderManualTable() {
    tblMan.innerHTML = "";
    for (const m of manualList) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${m.t != null ? Number(m.t).toFixed(4) : ""}</td><td>${escapeHtml(
        m.reason
      )}</td><td>${escapeHtml(m.label)}</td><td>${escapeHtml(
        m.in_clip_basename != null
          ? String(m.in_clip_basename)
          : m.note != null
            ? String(m.note)
            : ""
      )}</td>`;
      tblMan.appendChild(tr);
    }
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function loadPayload(data) {
    raw = data;
    sysS = deepCopyNumArr(data.start_times);
    sysT = deepCopyNumArr(data.stop_times);
    userS = deepCopyNumArr(data.start_times);
    userT = deepCopyNumArr(data.stop_times);
    durationSec = Number(data.duration_sec) || 0;
    manualList = Array.isArray(data.manual_review) ? data.manual_review : [];
    const tail = data.stop_beep_tail_sec;
    if (typeof tail === "number" && !Number.isNaN(tail) && tail >= 0) {
      previewTailSec = tail;
    } else {
      previewTailSec = 0.12;
    }
    document.getElementById("manualBlock").style.display = "block";
    fitView();
    renderTableS();
    renderTableT();
    renderManualTable();
    undoStack = [];
    redoStack = [];
  }

  function onMarkersFile(ev) {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = function () {
      try {
        const data = JSON.parse(r.result);
        loadPayload(data);
      } catch (e) {
        window.alert("JSON 解析失败: " + e);
      }
    };
    r.readAsText(f, "utf-8");
  }

  function onVideoFile(ev) {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    v.removeAttribute("src");
    v.src = URL.createObjectURL(f);
    v.load();
  }

  v.addEventListener("timeupdate", updatePlayhead);
  v.addEventListener("loadedmetadata", function () {
    if (!durationSec && v.duration && !Number.isNaN(v.duration)) {
      durationSec = v.duration;
    }
    clampView();
    updateViewReadout();
    updatePlayhead();
    redrawTimeline();
  });
  v.addEventListener("durationchange", function () {
    if (!raw || !(raw.duration_sec > 0)) {
      if (v.duration && !Number.isNaN(v.duration)) durationSec = v.duration;
    }
    clampView();
    updateViewReadout();
    updatePlayhead();
    redrawTimeline();
  });

  /** 表格单元格获得焦点时记一笔，便于一次撤销整段输入 */
  function onTableCellFocusIn(e) {
    const t = e.target;
    if (!t.matches || !t.matches("input.tcell")) {
      return;
    }
    recordBeforeChange();
  }
  tblS.addEventListener("focusin", onTableCellFocusIn);
  tblT.addEventListener("focusin", onTableCellFocusIn);

  function onCellInput(e) {
    const t = e.target;
    if (!t.matches || !t.matches("input.tcell")) return;
    const idx = parseInt(t.dataset.idx, 10);
    const x = parseFloat(String(t.value).replace(/,/g, "."));
    if (Number.isNaN(x)) {
      return;
    }
    if (t.dataset.k === "s" && idx >= 0 && idx < userS.length) {
      userS[idx] = x;
      redrawTimeline();
    } else if (t.dataset.k === "t" && idx >= 0 && idx < userT.length) {
      userT[idx] = x;
      redrawTimeline();
    }
  }
  tblS.addEventListener("input", onCellInput);
  tblT.addEventListener("input", onCellInput);
  tblS.addEventListener("change", function (e) {
    const t = e.target;
    if (t && t.matches && t.matches("input.tcell") && t.dataset.k === "s") {
      applyTableFromInputs();
      renderTableS();
      redrawTimeline();
    }
  });
  tblS.addEventListener("click", function (e) {
    if (e.target && e.target.dataset && e.target.dataset.act === "delS") {
      recordBeforeChange();
      const i = parseInt(e.target.dataset.i, 10);
      applyTableFromInputs();
      if (i >= 0 && i < userS.length) userS.splice(i, 1);
      userS.sort(sortNum);
      renderTableS();
      redrawTimeline();
    }
  });
  tblT.addEventListener("change", function (e) {
    const t = e.target;
    if (t && t.matches && t.matches("input.tcell") && t.dataset.k === "t") {
      applyTableFromInputs();
      renderTableT();
      redrawTimeline();
    }
  });
  tblT.addEventListener("click", function (e) {
    if (e.target && e.target.dataset && e.target.dataset.act === "delT") {
      recordBeforeChange();
      const i = parseInt(e.target.dataset.i, 10);
      applyTableFromInputs();
      if (i >= 0 && i < userT.length) userT.splice(i, 1);
      userT.sort(sortNum);
      renderTableT();
      redrawTimeline();
    }
  });

  document.getElementById("btnAddS").addEventListener("click", function () {
    recordBeforeChange();
    userS.push(v.currentTime || 0);
    userS.sort(sortNum);
    renderTableS();
    redrawTimeline();
  });
  document.getElementById("btnAddT").addEventListener("click", function () {
    recordBeforeChange();
    userT.push(v.currentTime || 0);
    userT.sort(sortNum);
    renderTableT();
    redrawTimeline();
  });
  document.getElementById("btnSortS").addEventListener("click", function () {
    recordBeforeChange();
    applyTableFromInputs();
    userS.sort(sortNum);
    renderTableS();
    redrawTimeline();
  });
  document.getElementById("btnSortT").addEventListener("click", function () {
    recordBeforeChange();
    applyTableFromInputs();
    userT.sort(sortNum);
    renderTableT();
    redrawTimeline();
  });

  document.getElementById("btnExport").addEventListener("click", function () {
    applyTableFromInputs();
    const base = raw
      ? JSON.parse(JSON.stringify(raw))
      : {
          file: "",
          duration_sec: durationSec,
          start_times: [],
          stop_times: [],
        };
    base.start_times = userS.slice();
    base.stop_times = userT.slice();
    if (v.duration && !Number.isNaN(v.duration) && v.duration > 0) {
      base.duration_sec = v.duration;
    } else if (raw && raw.duration_sec) {
      base.duration_sec = raw.duration_sec;
    } else {
      base.duration_sec = durationSec;
    }
    base.annotate_ui_edited_at = new Date().toISOString();
    const blob = new Blob([JSON.stringify(base, null, 2)], {
      type: "application/json; charset=utf-8",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "edited_markers.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  onlyUser.addEventListener("change", redrawTimeline);
  showPreview.addEventListener("change", redrawTimeline);

  fileMarkers.addEventListener("change", onMarkersFile);
  fileVideo.addEventListener("change", onVideoFile);

  window.addEventListener("keydown", function (e) {
    if (typingFocus()) return;
    const k = e.key;
    if ((e.metaKey || e.ctrlKey) && (k === "z" || k === "Z")) {
      e.preventDefault();
      if (e.shiftKey) {
        redoOnce();
      } else {
        undoOnce();
      }
      return;
    }
    if (e.ctrlKey && !e.metaKey && (k === "y" || k === "Y")) {
      e.preventDefault();
      redoOnce();
      return;
    }
    if (k === " " || k === "Spacebar") {
      e.preventDefault();
      if (v.paused) v.play();
      else v.pause();
      return;
    }
    if (k === "a" || k === "A") {
      e.preventDefault();
      recordBeforeChange();
      userS.push(v.currentTime || 0);
      userS.sort(sortNum);
      renderTableS();
      redrawTimeline();
      return;
    }
    if (k === "c" || k === "C") {
      e.preventDefault();
      recordBeforeChange();
      userT.push(v.currentTime || 0);
      userT.sort(sortNum);
      renderTableT();
      redrawTimeline();
      return;
    }
    if (k === "e" || k === "E") {
      e.preventDefault();
      if (!pickMarker) {
        return;
      }
      recordBeforeChange();
      if (!reclassifyPickAtSelection()) {
        undoStack.pop();
      } else {
        renderTableS();
        renderTableT();
        redrawTimeline();
      }
      return;
    }
  }, true);

  rail.addEventListener("pointerdown", function (e) {
    if (e.button !== 0) return;
    const mk = e.target.closest(
      ".layer.user .mk.userS, .layer.user .mk.userT"
    );
    if (mk && mk.dataset && mk.dataset.k !== undefined && mk.dataset.idx !== undefined) {
      pickMarker = {
        k: mk.dataset.k,
        idx: parseInt(mk.dataset.idx, 10),
      };
      redrawTimeline();
      return;
    }
    pickMarker = null;
    redrawTimeline();
    try {
      rail.setPointerCapture(e.pointerId);
    } catch (_err) {
      /*  */
    }
    railDrag = {
      pointerId: e.pointerId,
      x0: e.clientX,
      v0: viewStart,
      e0: viewEnd,
      moved: false,
    };
  });
  rail.addEventListener("pointermove", function (e) {
    if (!railDrag || e.pointerId !== railDrag.pointerId) return;
    const dx = e.clientX - railDrag.x0;
    if (Math.abs(dx) < 4) return;
    railDrag.moved = true;
    const r = rail.getBoundingClientRect();
    const span0 = railDrag.e0 - railDrag.v0;
    const maxD = getMaxD();
    if (span0 >= maxD - 1e-9) return;
    const dt = ((-(e.clientX - railDrag.x0)) / r.width) * span0;
    let ns = railDrag.v0 + dt;
    let ne = ns + span0;
    if (ns < 0) {
      ns = 0;
      ne = span0;
    }
    if (ne > maxD) {
      ne = maxD;
      ns = maxD - span0;
    }
    viewStart = ns;
    viewEnd = ne;
    clampView();
    updateViewReadout();
    redrawTimeline();
    updatePlayhead();
  });
  function railPointerUpOrCancel(e) {
    if (!railDrag || e.pointerId !== railDrag.pointerId) return;
    try {
      rail.releasePointerCapture(e.pointerId);
    } catch (_e2) {
      /*  */
    }
    if (!railDrag.moved && v.duration && !Number.isNaN(v.duration) && v.duration > 0) {
      const r = rail.getBoundingClientRect();
      const x = e.clientX - r.left;
      const ratio = r.width > 0 ? x / r.width : 0;
      v.currentTime = viewStart + Math.max(0, Math.min(1, ratio)) * getViewSpan();
      updatePlayhead();
    }
    railDrag = null;
  }
  rail.addEventListener("pointerup", railPointerUpOrCancel);
  rail.addEventListener("pointercancel", railPointerUpOrCancel);

  rail.addEventListener(
    "wheel",
    function (e) {
      e.preventDefault();
      const fac = e.deltaY < 0 ? 1 / ZOOM_WHEEL : ZOOM_WHEEL;
      const t = timeUnderPointer(e.clientX, rail);
      zoomAt(t, fac);
    },
    { passive: false }
  );

  document.getElementById("btnZoomIn").addEventListener("click", function () {
    const c = v && !Number.isNaN(v.currentTime)
      ? v.currentTime
      : (viewStart + viewEnd) / 2;
    zoomAt(c, 1 / ZOOM_FACTOR);
  });
  document.getElementById("btnZoomOut").addEventListener("click", function () {
    const c = v && !Number.isNaN(v.currentTime)
      ? v.currentTime
      : (viewStart + viewEnd) / 2;
    zoomAt(c, ZOOM_FACTOR);
  });
  document.getElementById("btnFit").addEventListener("click", function () {
    fitView();
  });
  document.getElementById("btnCenterPlay").addEventListener("click", function () {
    if (v.duration && !Number.isNaN(v.duration)) {
      viewCenteredOn(v.currentTime || 0);
    }
  });
  document.getElementById("panLeft").addEventListener("click", function () {
    panByTime(-PAN_STEP_FRACTION * getViewSpan());
  });
  document.getElementById("panRight").addEventListener("click", function () {
    panByTime(PAN_STEP_FRACTION * getViewSpan());
  });

  async function tryServerLoad() {
    if (location.protocol === "file:") return;
    const base = new URL(".", location.href);
    try {
      const r = await fetch(new URL("api/markers", base), { cache: "no-store" });
      if (!r.ok) return;
      const data = await r.json();
      loadPayload(data);
      v.src = new URL("stream/video", base).href;
      v.load();
    } catch (_) {
      /* 无服务时静默，用户可用文件选 */
    }
  }

  tryServerLoad();
})();
