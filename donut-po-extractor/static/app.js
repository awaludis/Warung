"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");
const state = { fd: null, samples: [], entry: null, sample: null, pollTimer: null, extraction: null };

// ---------- user / api ----------
function currentUser() {
  const n = $("#userName").value.trim();
  return `${n || "anon"} (${$("#userRole").value})`;
}
try {
  $("#userName").value = localStorage.getItem("po.user") || "";
  $("#userRole").value = localStorage.getItem("po.role") || "Data Annotator";
} catch (_) {}
for (const id of ["userName", "userRole"]) {
  $("#" + id).addEventListener("change", () => {
    try { localStorage.setItem("po.user", $("#userName").value); localStorage.setItem("po.role", $("#userRole").value); } catch (_) {}
  });
}

async function api(path, opts = {}) {
  const headers = { "X-User": currentUser(), ...(opts.headers || {}) };
  let body = opts.body;
  if (body && !(body instanceof FormData)) { headers["Content-Type"] = "application/json"; body = JSON.stringify(body); }
  let res;
  try { res = await fetch("/api" + path, { method: opts.method || "GET", headers, body }); }
  catch (_) { throw new Error("Tidak bisa menghubungi server aplikasi. Di Colab, jalankan ulang sel yang menyalakan aplikasi."); }
  const isJson = res.headers.get("content-type")?.includes("json");
  if (!res.ok && !isJson && [404, 502, 503, 504].includes(res.status)) {
    throw new Error(`Server aplikasi tidak menjawab (HTTP ${res.status}) — kemungkinan server berhenti atau runtime Colab terputus. Jalankan ulang sel yang menyalakan aplikasi, lalu muat ulang halaman. Data di Google Drive tetap aman.`);
  }
  const data = isJson ? await res.json() : await res.text();
  if (!res.ok) throw new Error(typeof data === "object" ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : data);
  return data;
}
function toast(msg, error = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (error ? " error" : "");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), error ? 7000 : 3000);
}
async function guard(btn, fn) {
  if (btn) btn.disabled = true;
  try { return await fn(); } catch (e) { toast(e.message, true); } finally { if (btn) btn.disabled = false; }
}

// ---------- tabs ----------
$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]"); if (!b) return;
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("hidden", s.id !== "tab-" + b.dataset.tab));
  ({ annotate: loadAnnotate, train: loadTrain, extract: loadExtract })[b.dataset.tab]();
});

async function loadStatus() {
  const s = await api("/status");
  const b = $("#modelBanner");
  const rel = s.model.release;
  if (!rel) { b.className = "banner"; b.textContent = "Belum ada model yang dirilis. Lengkapi anotasi (BUC 1) lalu training & rilis model (BUC 2)."; }
  else if (s.model.retraining_required) { b.className = "banner"; b.textContent = `Model ${rel.release_id} perlu retraining: ${s.model.reasons.join(" ")}`; }
  else { b.className = "banner ok"; b.textContent = `Model aktif: ${rel.release_id} · akurasi field ${pct(rel.validation_summary.field_accuracy)} · layout: ${rel.trained_layouts.join(", ")}`; }
  $("#baseModel").placeholder = s.base_model;
  $("#trainedLayouts").innerHTML = (rel?.trained_layouts || []).map((l) => `<option value="${esc(l)}">`).join("");
  return s;
}

// =================== BUC 1 ===================
async function loadAnnotate() {
  const [samples, cov, records] = await Promise.all([api("/samples"), api("/coverage"), api("/gt-records")]);
  state.samples = samples;
  $("#layoutList").innerHTML = Object.keys(cov.layouts).map((l) => `<option value="${esc(l)}">`).join("");
  const rows = Object.entries(cov.layouts);
  $("#coverage").innerHTML = rows.length ? `<table><tr><th>Customer layout</th><th class="num">Selesai</th><th class="num">Draft</th><th>Status</th></tr>${rows.map(([l, r]) =>
    `<tr><td>${esc(l)}</td><td class="num">${r.completed}</td><td class="num">${r.draft}</td><td>${r.outdated ? `<span class="badge bad">${r.outdated} perlu disimpan ulang</span>` : r.ready ? '<span class="badge ok">siap</span>' : `<span class="badge warn">min ${cov.min_per_layout} selesai</span>`}</td></tr>`).join("")}</table>`
    : '<p class="muted">Belum ada sampel.</p>';
  $("#records").innerHTML = records.length ? `<h2 style="margin-top:14px">PO Ground Truth Record (filed)</h2><table><tr><th>Record</th><th>Filed</th><th class="num">Halaman</th><th>Layout</th></tr>${records.map((r) =>
    `<tr><td>${esc(r.record_id)}</td><td>${esc(r.filed_at)}<br><small class="muted">${esc(r.filed_by)}</small></td><td class="num">${r.entry_count}</td><td>${Object.entries(r.layouts).map(([l, n]) => `${esc(l)} (${n})`).join(", ")}</td></tr>`).join("")}</table>` : "";
  $("#samples").innerHTML = samples.length ? samples.map((s) => `
    <div class="sample">
      <div class="meta"><b>${esc(s.filename)}</b> <span class="badge">${esc(s.customer_layout)}</span>
        <small>${esc(s.sample_id)} · ${s.completed_pages}/${s.pages.length} halaman selesai</small></div>
      ${s.pages.map((p) => `<button class="small page-btn${p.status === "completed" ? " completed" : ""}${state.entry?.entry_id === p.entry_id ? " current" : ""}" data-entry="${esc(p.entry_id)}" data-sample="${esc(s.sample_id)}">Hal. ${p.page}</button>`).join("")}
      <button class="small" data-del="${esc(s.sample_id)}" title="Hapus sampel">Hapus</button>
    </div>`).join("") : '<p class="muted">Unggah sampel PO untuk mulai anotasi.</p>';
  loadStatus();
}

$("#sampleForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const btn = e.submitter;
  guard(btn, async () => {
    btn.textContent = "Mengonversi…";
    try {
      const s = await api("/samples", { method: "POST", body: new FormData(e.target) });
      toast(`Sampel diunggah: ${s.pages.length} halaman.`);
      e.target.reset();
      await loadAnnotate();
      openEntry(s.pages[0].entry_id, s.sample_id);
    } finally { btn.textContent = "Unggah & konversi ke gambar"; }
  });
});

$("#samples").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.entry) openEntry(b.dataset.entry, b.dataset.sample);
  if (b.dataset.del && confirm("Hapus sampel ini beserta anotasinya?")) guard(b, async () => {
    await api("/samples/" + b.dataset.del, { method: "DELETE" });
    if (state.sample?.sample_id === b.dataset.del) { $("#annotator").classList.add("hidden"); state.entry = null; }
    loadAnnotate();
  });
});

function fieldInput(f, value, path) {
  const type = f.type === "number" ? "text\" inputmode=\"decimal" : "text";
  return `<input type="${type}" data-path="${esc(path)}" value="${esc(value)}" placeholder="${f.type === "date" ? "seperti tertulis di PO" : ""}">`;
}
function rowHtml(f, row, idx) {
  return `<tr>${f.fields.map((s) => `<td>${fieldInput(s, row[s.name], `${f.name}.${idx}.${s.name}`)}</td>`).join("")}<td><button type="button" class="small" data-remove-row>✕</button></td></tr>`;
}
function renderForm(values) {
  $("#annotForm").innerHTML = state.fd.fields.map((f) => {
    if (f.type !== "list") return `<label><span class="${f.required ? "req" : ""}">${esc(f.label)} <small>${esc(f.name)} · ${f.type}</small></span>${fieldInput(f, values[f.name], f.name)}</label>`;
    const rows = values[f.name]?.length ? values[f.name] : [{}];
    return `<div class="list-field" data-list="${esc(f.name)}"><div class="row between"><b class="${f.required ? "req" : ""}">${esc(f.label)}</b><button type="button" class="small" data-add-row="${esc(f.name)}">+ Baris</button></div>
      <div class="scroll"><table><tr>${f.fields.map((s) => `<th class="${s.required ? "req" : ""}">${esc(s.label)}</th>`).join("")}<th></th></tr>${rows.map((r, i) => rowHtml(f, r, i)).join("")}</table></div></div>`;
  }).join("");
}
$("#annotForm").addEventListener("click", (e) => {
  const add = e.target.closest("[data-add-row]");
  if (add) {
    const f = state.fd.fields.find((x) => x.name === add.dataset.addRow);
    const table = add.closest(".list-field").querySelector("table");
    table.insertAdjacentHTML("beforeend", rowHtml(f, {}, table.rows.length - 1));
  }
  if (e.target.closest("[data-remove-row]")) e.target.closest("tr").remove();
});
function collectForm() {
  const values = {};
  for (const f of state.fd.fields) {
    if (f.type !== "list") { values[f.name] = $(`[data-path="${f.name}"]`, $("#annotForm")).value; continue; }
    const rows = [];
    $(`[data-list="${f.name}"]`).querySelectorAll("tr").forEach((tr) => {
      const inputs = tr.querySelectorAll("input[data-path]"); if (!inputs.length) return;
      const row = {}; inputs.forEach((i) => { row[i.dataset.path.split(".").pop()] = i.value; });
      if (Object.values(row).some((v) => v.trim())) rows.push(row);
    });
    values[f.name] = rows;
  }
  return values;
}

async function openEntry(entryId, sampleId) {
  const [entry, sample] = await Promise.all([api("/entries/" + entryId), api("/samples/" + sampleId)]);
  state.entry = entry; state.sample = sample;
  $("#annotator").classList.remove("hidden");
  $("#annotTitle").textContent = `${sample.filename} · halaman ${entry.page}/${sample.pages.length} · ${entry.customer_layout}`;
  $("#annotImg").src = `/api/entries/${entryId}/image`;
  $("#annotStatus").className = "badge " + (entry.status === "completed" ? "ok" : "");
  $("#annotStatus").textContent = entry.status === "completed" ? "selesai" : "draft";
  renderForm(entry.values);
  document.querySelectorAll(".page-btn").forEach((b) => b.classList.toggle("current", b.dataset.entry === entryId));
  $("#prevPage").disabled = entry.page <= 1;
  $("#nextPage").disabled = entry.page >= sample.pages.length;
  $("#annotator").scrollIntoView({ behavior: "smooth", block: "start" });
}
function gotoPage(delta) {
  const p = state.sample.pages.find((x) => x.page === state.entry.page + delta);
  if (p) openEntry(p.entry_id, state.sample.sample_id);
}
$("#prevPage").onclick = () => gotoPage(-1);
$("#nextPage").onclick = () => gotoPage(1);
async function saveEntry(complete, btn) {
  await guard(btn, async () => {
    const e = await api("/entries/" + state.entry.entry_id, { method: "PUT", body: { values: collectForm(), complete } });
    toast(complete ? "Ground truth entry disimpan & ditandai selesai." : "Draft disimpan.");
    await loadAnnotate();
    if (complete && e.page < state.sample.pages.length) gotoPage(1); else openEntry(e.entry_id, e.sample_id);
  });
}
$("#saveDraft").onclick = (e) => saveEntry(false, e.target);
$("#saveComplete").onclick = (e) => saveEntry(true, e.target);
$("#importPractice").onclick = (e) => {
  if (!confirm("Impor 45 dokumen latihan beserta kunci jawabannya? Proses ±1 menit.")) return;
  const btn = e.target;
  guard(btn, async () => {
    btn.textContent = "Mengimpor…";
    try {
      const r = await api("/practice/import", { method: "POST", body: { draft: $("#practiceDraft").checked } });
      toast(`${r.documents} dokumen / ${r.pages} halaman diimpor. Lanjut: File Ground Truth Record.`);
      loadAnnotate();
    } finally { btn.textContent = "Impor dokumen latihan"; }
  });
};
$("#fileRecordBtn").onclick = (e) => guard(e.target, async () => {
  const r = await api("/gt-records", { method: "POST", body: { note: $("#recordNote").value } });
  toast(`PO Ground Truth Record ${r.record_id} di-file (${r.entry_count} halaman).`);
  $("#recordNote").value = "";
  loadAnnotate();
});

// =================== BUC 2 ===================
const STATUS_BADGE = { created: "", training: "warn", trained: "", validating: "warn", validated: "ok", failed: "bad", cancelled: "bad" };
async function loadTrain() {
  const [records, runs, releases] = await Promise.all([api("/gt-records"), api("/runs"), api("/releases")]);
  $("#recordSelect").innerHTML = records.length ? records.map((r) => `<option value="${esc(r.record_id)}">${esc(r.record_id)} · ${r.entry_count} halaman · ${Object.keys(r.layouts).length} layout</option>`).join("")
    : '<option value="">Belum ada record — selesaikan BUC 1</option>';
  $("#runs").innerHTML = runs.length ? `<div class="scroll"><table><tr><th>Run</th><th>Record</th><th>Status</th><th>Progres</th><th class="num">Akurasi field</th><th class="num">Exact match</th><th></th></tr>${runs.map((r) => `
    <tr><td>${esc(r.run_id)}<br><small class="muted">${esc(r.created_by)}</small></td><td>${esc(r.record_id)}</td>
    <td><span class="badge ${STATUS_BADGE[r.status] || ""}">${esc(r.status)}</span>${r.released ? ' <span class="badge ok">released</span>' : ""}</td>
    <td><div class="progress"><div style="width:${(r.progress || 0) * 100}%"></div></div></td>
    <td class="num">${pct(r.validation_summary?.field_accuracy)}</td><td class="num">${pct(r.validation_summary?.document_exact_match)}</td>
    <td><button class="small" data-run="${esc(r.run_id)}">Detail</button></td></tr>`).join("")}</table></div>` : '<p class="muted">Belum ada training run.</p>';
  $("#releases").innerHTML = releases.length ? `<table><tr><th>Release</th><th>Run</th><th class="num">Akurasi</th><th>Layout</th></tr>${releases.map((r, i) => `
    <tr><td>${esc(r.release_id)} ${i === 0 ? '<span class="badge ok">aktif</span>' : ""}<br><small class="muted">${esc(r.released_at)} · ${esc(r.released_by)}</small></td><td>${esc(r.run_id)}</td>
    <td class="num">${pct(r.validation_summary.field_accuracy)}</td><td>${r.trained_layouts.map(esc).join(", ")}</td></tr>`).join("")}</table>` : '<p class="muted">Belum ada model rilis.</p>';
  loadStatus();
  clearTimeout(state.pollTimer);
  if (runs.some((r) => r.status === "training" || r.status === "validating")) {
    state.pollTimer = setTimeout(() => { if (!$("#tab-train").classList.contains("hidden")) { loadTrain(); if (state.runId) showRun(state.runId, false); } }, 3000);
  }
}
$("#runForm").addEventListener("submit", (e) => {
  e.preventDefault();
  guard(e.submitter, async () => {
    const fd = new FormData(e.target), params = {};
    for (const k of ["validation_ratio", "learning_rate"]) params[k] = parseFloat(fd.get(k));
    for (const k of ["epochs", "batch_size", "image_height", "image_width"]) if (fd.get(k)) params[k] = parseInt(fd.get(k), 10);
    if (fd.get("base_model")) params.base_model = fd.get("base_model");
    params.auto_validate = !!fd.get("auto_validate");
    params.gradient_checkpointing = !!fd.get("gradient_checkpointing");
    const run = await api("/runs", { method: "POST", body: { record_id: fd.get("record_id"), params } });
    toast(`Run ${run.run_id}: ${run.split.train.length} halaman training / ${run.split.validation.length} validasi.`);
    await loadTrain(); showRun(run.run_id);
  });
});
$("#runs").addEventListener("click", (e) => { const b = e.target.closest("[data-run]"); if (b) showRun(b.dataset.run); });

async function showRun(runId, scroll = true) {
  state.runId = runId;
  const run = await api("/runs/" + runId);
  let sheet = null;
  if (run.status === "validated") sheet = await api(`/runs/${runId}/validation-sheet`);
  const layouts = Object.entries(run.layouts).map(([l, c]) => `${esc(l)}: ${c.train} train / ${c.validation} val`).join(" · ");
  const el = $("#runDetail");
  el.classList.remove("hidden");
  el.innerHTML = `
    <div class="row between"><h2>${esc(run.run_id)} <span class="badge ${STATUS_BADGE[run.status] || ""}">${esc(run.status)}</span></h2>
      <div class="row wrap">
        ${["created", "failed", "cancelled"].includes(run.status) ? '<button data-act="train">Mulai training</button>' : ""}
        ${run.status === "training" ? '<button data-act="cancel">Batalkan</button>' : ""}
        ${["trained", "validated"].includes(run.status) ? '<button data-act="validate">Jalankan validasi</button>' : ""}
        ${sheet ? `<a href="/api/runs/${esc(runId)}/validation-sheet?format=csv"><button>Unduh Validation Sheet (CSV)</button></a>` : ""}
      </div></div>
    <p class="muted">Record ${esc(run.record_id)} · ${layouts}<br>Parameter: ${esc(JSON.stringify(run.params))}</p>
    ${run.error ? `<p class="badge bad">${esc(run.error)}</p>` : ""}
    ${sheet ? sheetHtml(sheet) : ""}
    ${sheet && !run.released ? `<div class="row wrap end"><input id="relNote" placeholder="Catatan rilis"><input id="relMin" type="number" step="0.01" min="0" max="1" placeholder="Min. akurasi (opsional, 0–1)" style="max-width:220px"><button class="primary" data-act="release">Rilis model</button></div>` : ""}
    <h2 style="margin-top:14px">Log</h2><pre class="log">${esc(run.log || "")}</pre>`;
  el.onclick = (e) => {
    const b = e.target.closest("[data-act]"); if (!b) return;
    guard(b, async () => {
      const act = b.dataset.act;
      if (act === "release") {
        const min = $("#relMin").value;
        const rel = await api(`/runs/${runId}/release`, { method: "POST", body: { note: $("#relNote").value, min_field_accuracy: min ? parseFloat(min) : null } });
        toast(`Model dirilis: ${rel.release_id}`);
      } else {
        await api(`/runs/${runId}/${act}`, { method: "POST" });
      }
      await loadTrain(); showRun(runId, false);
    });
  };
  if (scroll) el.scrollIntoView({ behavior: "smooth", block: "start" });
}
function sheetHtml(s) {
  const layouts = Object.keys(s.per_layout);
  const cell = (c) => (c && c.total ? `${pct(c.accuracy)} <small class="muted">~${pct(c.similarity)}</small>` : "–");
  const rows = Object.entries(s.per_field).map(([k, c]) => `<tr><td>${esc(k)}</td><td class="num">${cell(c)} <small class="muted">${c.correct}/${c.total}</small></td>${layouts.map((l) => `<td class="num">${cell(s.per_layout[l].fields[k])}</td>`).join("")}</tr>`).join("");
  const mism = s.details.filter((d) => !d.exact_match);
  return `<h2>Model Validation Sheet</h2>
    <p>Halaman validasi: <b>${s.validation_pages}</b> · Akurasi field (persis sama): <b>${pct(s.overall.field_accuracy)}</b> · Kemiripan karakter: <b>${pct(s.overall.similarity)}</b> · Exact match dokumen: <b>${pct(s.overall.document_exact_match)}</b> · ${esc(s.validated_at)} oleh ${esc(s.validated_by)}</p>
    <p class="muted">Angka kecil "~" = kemiripan karakter rata-rata: tinggi tetapi akurasi rendah berarti model hampir benar (salah 1–2 karakter) — biasanya membaik dengan lebih banyak data / epoch.</p>
    <div class="scroll"><table><tr><th>Field</th><th class="num">Semua layout</th>${layouts.map((l) => `<th class="num">${esc(l)}</th>`).join("")}</tr>
    <tr><td><b>Total</b></td><td class="num"><b>${pct(s.overall.field_accuracy)}</b> <small class="muted">~${pct(s.overall.similarity)}</small></td>${layouts.map((l) => `<td class="num"><b>${pct(s.per_layout[l].field_accuracy)}</b> <small class="muted">~${pct(s.per_layout[l].similarity)}</small></td>`).join("")}</tr>${rows}</table></div>
    ${mism.length ? `<details style="margin-top:10px"><summary>${mism.length} halaman dengan selisih</summary><div class="scroll"><table><tr><th>Entry</th><th>Layout</th><th>Field berbeda</th></tr>${mism.map((d) => `<tr><td>${esc(d.entry_id)}</td><td>${esc(d.customer_layout)}</td><td>${d.mismatched_fields.map(esc).join(", ")}</td></tr>`).join("")}</table></div></details>` : ""}`;
}

// =================== BUC 3 ===================
async function loadExtract() {
  await loadStatus();
  const list = await api("/extractions");
  $("#extractions").innerHTML = list.length ? `<div class="scroll"><table><tr><th>Ekstraksi</th><th>File</th><th>PO number</th><th>Layout</th><th>Status</th><th></th></tr>${list.map((x) => `
    <tr><td>${esc(x.extraction_id)}<br><small class="muted">${esc(x.extracted_at)}</small></td><td>${esc(x.filename)} <small class="muted">(${x.pages} hal.)</small></td><td>${esc(x.po_number ?? "–")}</td><td>${esc(x.customer_layout || "–")}</td>
    <td><span class="badge ${x.status === "saved" ? "ok" : ""}">${esc(x.status)}</span></td>
    <td><button class="small" data-open="${esc(x.extraction_id)}">Buka</button>${x.status === "saved" ? ` <a href="/api/extractions/${esc(x.extraction_id)}/json"><button class="small">JSON</button></a>` : ""}</td></tr>`).join("")}</table></div>` : '<p class="muted">Belum ada ekstraksi.</p>';
}
$("#extractions").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-open]"); if (b) guard(b, async () => showExtraction(await api("/extractions/" + b.dataset.open)));
});
$("#extractForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const btn = e.submitter;
  guard(btn, async () => {
    btn.textContent = "Membaca halaman…";
    try {
      const res = await api("/extractions", { method: "POST", body: new FormData(e.target) });
      showExtraction(res); loadExtract();
    } finally { btn.textContent = "Ekstrak"; }
  });
});
function showExtraction(x) {
  state.extraction = x;
  const el = $("#extractResult");
  el.classList.remove("hidden");
  el.innerHTML = `
    <div class="row between"><h2>${esc(x.filename)} <span class="badge ${x.status === "saved" ? "ok" : ""}">${esc(x.status)}</span></h2>
      <small class="muted">${esc(x.extraction_id)} · model ${esc(x.release_id)}</small></div>
    ${x.warnings.length ? `<ul class="warnings">${x.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
    <div class="thumbs">${x.pages.map((p) => `<a href="/api/extractions/${esc(x.extraction_id)}/${esc(p.image)}" target="_blank"><img src="/api/extractions/${esc(x.extraction_id)}/${esc(p.image)}" alt="Halaman ${p.page}"></a>`).join("")}</div>
    <details style="margin-top:10px"><summary>Output mentah model per halaman (untuk diagnosis)</summary>
      ${x.pages.map((p) => `<p class="muted">Halaman ${p.page}</p><pre class="log">${esc(p.raw_sequence || "(kosong — model tidak menghasilkan apa pun)")}</pre>`).join("")}
    </details>
    <label style="margin-top:12px">Structured PO data (JSON) — dapat dikoreksi sebelum disimpan
      <textarea id="poJson" rows="22" spellcheck="false">${esc(JSON.stringify(x.data, null, 2))}</textarea></label>
    <div class="row end">
      <button id="saveExtraction" class="primary">Simpan PO data</button>
      ${x.status === "saved" ? `<a href="/api/extractions/${esc(x.extraction_id)}/json"><button>Unduh JSON</button></a>` : ""}
    </div>`;
  $("#saveExtraction").onclick = (e) => guard(e.target, async () => {
    let data;
    try { data = JSON.parse($("#poJson").value); } catch (err) { throw new Error("JSON tidak valid: " + err.message); }
    const saved = await api(`/extractions/${x.extraction_id}/save`, { method: "POST", body: { data } });
    toast("Structured PO Data Record disimpan.");
    showExtraction(saved); loadExtract();
  });
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- boot ----------
(async () => {
  try {
    state.fd = await api("/field-definition");
    if ((await api("/practice")).available) $("#practiceBox").classList.remove("hidden");
    await loadAnnotate();
  } catch (e) { toast(e.message, true); }
})();
