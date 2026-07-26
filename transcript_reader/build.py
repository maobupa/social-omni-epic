#!/usr/bin/env python3
"""Build a self-contained HTML reader for the Gen-90 curriculum transcripts.

Reads the per-scenario generated JSONs from one or more curriculum runs, extracts
the transcripts / retries / reflexions / scores / categories, and emits a single
offline HTML file you double-click to open.

    uv run transcript_reader/build.py           # builds transcript_reader/reader.html

Categories (from classification + terminal_success — see docs):
    too_easy            solved on attempt 1
    frontier_solved     LP>0, eventually solved after retries
    frontier_unsolved   LP>0 (learner improved) but never solved
    beyond_frontier     LP=0, no learning progress

Run this from the repo root (it reads results/gen90_expel* by relative path).
"""
import json
import sys
from pathlib import Path

# run label -> results dir
RUNS = {
    "gpt-5-mini": "results/gen90_expel",
    "gpt-4.1-mini": "results/gen90_expel_41mini",
}
OUT = Path(__file__).parent / "reader.html"

CATEGORY_ORDER = ["too_easy", "frontier_solved", "frontier_unsolved", "beyond_frontier"]


def category_of(classification: str, solved: bool) -> str:
    if classification == "too_easy":
        return "too_easy"
    if classification == "frontier":
        return "frontier_solved" if solved else "frontier_unsolved"
    return "beyond_frontier"


def agent_name(profile) -> str:
    if not isinstance(profile, dict):
        return str(profile)
    fn = profile.get("first_name", "") or ""
    ln = profile.get("last_name", "") or ""
    return (fn + " " + ln).strip() or "Agent"


# 7 Sotopia dims; goal & relationship are the success-defining ones.
DIMS = ["goal", "relationship", "believability", "knowledge",
        "secret", "social_rules", "financial_and_material_benefits"]


def extract_scenario(d: dict) -> dict:
    profiles = d.get("agent_profiles") or []
    tidx = d.get("target_agent_idx", 0) or 0
    learner = agent_name(profiles[tidx]) if tidx < len(profiles) else "Learner"
    partner = agent_name(profiles[1 - tidx]) if len(profiles) > 1 else "Partner"
    goals = d.get("agent_goals") or []

    solved = bool(d.get("terminal_success"))
    classification = d.get("classification") or "unknown"

    attempts = []
    for a in d.get("attempts", []):
        transcript = [
            {"turn": t.get("turn"), "speaker": t.get("speaker", "?"),
             "content": t.get("content", "")}
            for t in (a.get("transcript") or [])
        ]
        scores = a.get("scores") or {}
        attempts.append({
            "attempt": a.get("attempt"),
            "solved": bool(a.get("solved")),
            "scores": {k: scores.get(k) for k in DIMS},
            "key_check": a.get("key_check_result"),
            "reflexion": a.get("reflexion") or "",
            "transcript": transcript,
        })

    return {
        "id": d.get("id"),
        "title": d.get("scenario_title") or (d.get("scenario", "")[:60] + "…"),
        "scenario": d.get("scenario", ""),
        "learner": learner,
        "partner": partner,
        "learner_goal": goals[tidx] if tidx < len(goals) else "",
        "partner_goal": goals[1 - tidx] if len(goals) > 1 else "",
        "rubric": d.get("success_rubric", ""),
        "classification": classification,
        "solved": solved,
        "category": category_of(classification, solved),
        "lp_value": d.get("lp_value"),
        "mutation_operator": d.get("mutation_operator"),
        "lineage_depth": d.get("lineage_depth"),
        "source": d.get("source"),
        "root_seed": d.get("root_seed_env_pk"),
        "chronicle": d.get("skills_final_md") or "",
        "goal_traj": [(a.get("scores") or {}).get("goal") for a in d.get("attempts", [])],
        "attempts": attempts,
    }


def load_run(results_dir: str) -> dict:
    root = Path(results_dir)
    gen = sorted((root / "bank" / "generated").glob("*.json"))
    scenarios = []
    for f in gen:
        try:
            scenarios.append(extract_scenario(json.loads(f.read_text())))
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)
    # global insights (the extracted ExpeL rules)
    insights = []
    ip = root / "insights.json"
    if ip.exists():
        try:
            obj = json.loads(ip.read_text())
            raw = obj.get("all", obj) if isinstance(obj, dict) else obj
            # insights['all'] is {"rules": [...], "rule_items_with_count": [...]}
            if isinstance(raw, dict):
                raw = raw.get("rules") or raw.get("all") or []
            insights = [str(r) for r in (raw or [])]
        except Exception as e:
            print(f"  insights load failed for {results_dir}: {e}", file=sys.stderr)
    scenarios.sort(key=lambda s: (CATEGORY_ORDER.index(s["category"])
                                  if s["category"] in CATEGORY_ORDER else 99, s["title"]))
    return {"scenarios": scenarios, "insights": insights}


def main():
    data = {}
    for label, d in RUNS.items():
        if Path(d).exists():
            data[label] = load_run(d)
            n = len(data[label]["scenarios"])
            print(f"{label:14} {d}: {n} scenarios, {len(data[label]['insights'])} insights")
        else:
            print(f"{label:14} {d}: (missing — skipped)")
    if not data:
        print("No runs found. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"Open it: open {OUT}")


# --------------------------------------------------------------------------- #
# Self-contained HTML/CSS/JS.  __DATA__ is replaced with the JSON payload.
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gen-90 Transcript Reader</title>
<style>
:root{
  --bg:#faf9f7; --panel:#fff; --ink:#1e1c1a; --muted:#8a8580; --line:#e6e2dc;
  --learner:#2563a8; --learner-bg:#eaf2fb; --partner:#8a6d3b; --partner-bg:#f6f0e6;
  --reflex-bg:#fff8e1; --reflex-bd:#e9c46a;
  --too_easy:#9aa0a6; --frontier_solved:#2e8b57; --frontier_unsolved:#d98a00; --beyond_frontier:#c0392b;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px}
.app{display:flex;height:100vh;overflow:hidden}
/* sidebar */
.sidebar{width:320px;flex:0 0 320px;background:var(--panel);border-right:1px solid var(--line);
  display:flex;flex-direction:column;transition:width .15s,flex-basis .15s}
.sidebar.collapsed{width:0;flex-basis:0;overflow:hidden}
.sb-head{padding:10px 12px;border-bottom:1px solid var(--line)}
.runtoggle{display:flex;gap:4px;margin-bottom:8px}
.runtoggle button{flex:1;padding:6px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer;font-size:12px}
.runtoggle button.active{background:var(--ink);color:#fff;border-color:var(--ink)}
.search{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;font-size:13px}
.catfilter{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.catchip{font-size:11px;padding:2px 7px;border-radius:10px;cursor:pointer;border:1px solid var(--line);user-select:none;opacity:.45}
.catchip.on{opacity:1}
.sb-list{overflow-y:auto;flex:1}
.grp-h{position:sticky;top:0;background:var(--bg);padding:5px 12px;font-size:11px;font-weight:700;
  text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--line)}
.item{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:8px;align-items:center}
.item:hover{background:var(--bg)}
.item.sel{background:var(--learner-bg)}
.item .ttl{flex:1;font-size:12.5px;line-height:1.3}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 8px}
.item .meta{font-size:10.5px;color:var(--muted);white-space:nowrap}
/* main */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}
.topbar{display:flex;align-items:center;gap:10px;padding:8px 12px;border-bottom:1px solid var(--line);background:var(--panel)}
.iconbtn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:5px 9px;cursor:pointer;font-size:13px}
.header{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
.h-title{font-size:17px;font-weight:700;margin:0 0 4px}
.badges{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.badge{font-size:11px;padding:2px 9px;border-radius:10px;color:#fff;font-weight:600}
.kv{font-size:12.5px;color:#444;margin:3px 0}
.kv b{color:var(--ink)}
.goalrow{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--muted)}
/* columns */
.cols{flex:1;overflow:auto;display:flex;gap:12px;padding:12px 16px;align-items:flex-start}
.col{flex:0 0 auto;width:380px;min-width:240px;max-width:1200px;background:var(--panel);border:1px solid var(--line);border-radius:8px;
  display:flex;flex-direction:column;max-height:100%;resize:horizontal;overflow:hidden}
.col-h{padding:8px 10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);border-radius:8px 8px 0 0}
.col-h .a-title{font-weight:700;font-size:13px;display:flex;justify-content:space-between;align-items:center}
.chips{display:flex;gap:4px;flex-wrap:wrap;margin-top:5px}
.chip{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);background:#fff}
.chip b{font-variant-numeric:tabular-nums}
.turns{overflow-y:auto;padding:8px 8px}
.turn{margin:5px 0;padding:6px 9px;border-radius:8px;font-size:13px;line-height:1.42}
.turn.learner{background:var(--learner-bg);border-left:3px solid var(--learner)}
.turn.partner{background:var(--partner-bg);border-left:3px solid var(--partner)}
.turn .sp{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;margin-bottom:2px}
.turn.learner .sp{color:var(--learner)} .turn.partner .sp{color:var(--partner)}
.turn .sp .tn{color:var(--muted);font-weight:500;margin-left:6px}
.turn.act{background:transparent;border-left:3px solid var(--line);font-style:italic;color:var(--muted)}
.reflex{margin:8px;background:var(--reflex-bg);border:1px solid var(--reflex-bd);border-radius:8px;font-size:12.5px;line-height:1.4}
.reflex>summary{cursor:pointer;padding:6px 10px;font-weight:700;font-size:11px;color:#b9821f;list-style:none;user-select:none}
.reflex>summary::-webkit-details-marker{display:none}
.reflex>summary::before{content:"▸ ";}
.reflex[open]>summary::before{content:"▾ ";}
.reflex .reflex-body{padding:0 10px 8px}
/* collapsible scenario header */
details.header{padding:0}
.header>summary{cursor:pointer;padding:12px 16px;list-style:none;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.header>summary::-webkit-details-marker{display:none}
.header>summary::before{content:"▾";color:var(--muted);font-size:12px}
.header:not([open])>summary::before{content:"▸"}
.hbody{padding:0 16px 12px;overflow:auto;resize:vertical}
/* AI panel (in drawer) */
/* Ask-AI chat */
.chat{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:9px}
.msg{padding:8px 11px;border-radius:11px;font-size:13px;line-height:1.5;white-space:pre-wrap;max-width:92%;word-wrap:break-word}
.msg.user{align-self:flex-end;background:var(--learner-bg);border:1px solid #cfe0f3}
.msg.ai{align-self:flex-start;background:var(--bg);border:1px solid var(--line)}
.msg.pending{opacity:.55;font-style:italic}
.chat-empty{margin:auto;color:var(--muted);text-align:center;font-size:12.5px;padding:20px;line-height:1.5}
.chat-input{border-top:1px solid var(--line);padding:8px 10px;background:var(--panel)}
.chat-input .hint{font-size:11px;color:var(--muted);margin-bottom:6px;line-height:1.4}
.chat-input .hint code{background:#eee;padding:1px 4px;border-radius:4px}
.chat-input textarea{width:100%;min-height:42px;max-height:150px;padding:7px 9px;border:1px solid var(--line);border-radius:8px;font:inherit;font-size:13px;resize:vertical;box-sizing:border-box}
.aistatus{font-size:10.5px;padding:1px 7px;border-radius:9px;margin-left:8px}
.aistatus.up{background:#e4f5ec;color:#1d7a45} .aistatus.down{background:#fdecea;color:#c0392b}
/* global-insights modal (run-level, NOT per scenario) */
.modal{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;align-items:center;justify-content:center;z-index:20}
.modal.open{display:flex}
.modal-box{background:var(--panel);width:min(720px,92%);max-height:84vh;border-radius:12px;display:flex;flex-direction:column;box-shadow:0 14px 44px rgba(0,0,0,.28)}
.modal-h{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line)}
.modal-note{padding:10px 16px;background:var(--reflex-bg);border-bottom:1px solid var(--reflex-bd);font-size:12.5px;line-height:1.45}
.modal-body{overflow-y:auto;padding:6px 16px 14px}
.solvedtag{font-size:11px;font-weight:700;padding:1px 8px;border-radius:9px}
.solved{background:#e4f5ec;color:#1d7a45} .unsolved{background:#fdecea;color:#c0392b}
/* drawer */
.drawer{position:absolute;top:0;right:0;height:100%;width:min(460px,42%);background:var(--panel);
  border-left:1px solid var(--line);box-shadow:-6px 0 20px rgba(0,0,0,.08);transform:translateX(100%);
  transition:transform .18s;display:flex;flex-direction:column;z-index:5}
.drawer.open{transform:translateX(0)}
.drawer-h{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line)}
.drawer-body{overflow-y:auto;padding:14px}
.drawer h4{margin:14px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.chronicle{white-space:pre-wrap;font-size:13px;line-height:1.5;background:var(--bg);padding:10px;border-radius:8px;border:1px solid var(--line)}
.insight{font-size:12.5px;line-height:1.45;padding:6px 8px;border-bottom:1px solid var(--line)}
.attctl{display:flex;gap:6px;align-items:center;margin-left:auto;font-size:12px;color:var(--muted)}
.attctl label{cursor:pointer;user-select:none}
.empty{margin:auto;color:var(--muted);text-align:center;padding:40px}
.spark{display:inline-block;vertical-align:middle}
</style></head>
<body>
<div class="app">
  <aside class="sidebar" id="sidebar">
    <div class="sb-head">
      <div class="runtoggle" id="runtoggle"></div>
      <input class="search" id="search" placeholder="Search title / scenario…">
      <div class="catfilter" id="catfilter"></div>
    </div>
    <div class="sb-list" id="list"></div>
  </aside>
  <main class="main">
    <div class="topbar">
      <button class="iconbtn" id="collapse" title="Toggle list">☰</button>
      <div id="crumb" style="font-size:12px;color:var(--muted)"></div>
      <div class="attctl" id="attctl"></div>
      <button class="iconbtn" id="insbtn" title="Global insights for this run (same for all scenarios)" style="margin-left:8px">🌐 Insights</button>
      <button class="iconbtn" id="drawerbtn" title="Ask AI about this scenario">🤖 Ask AI</button>
    </div>
    <div id="content" style="flex:1;display:flex;flex-direction:column;overflow:hidden">
      <div class="empty">Select a scenario from the list.</div>
    </div>
    <aside class="drawer" id="drawer">
      <div class="drawer-h"><b>🤖 Ask AI</b><span class="aistatus" id="aistatus">…</span>
        <span style="flex:1"></span>
        <button class="iconbtn" id="aiclear" title="Clear conversation" style="font-size:11px;padding:3px 8px">Clear</button>
        <button class="iconbtn" id="drawerclose">✕</button></div>
      <div class="chat" id="chat"></div>
      <div class="chat-input">
        <div class="hint">💬 Conversation persists as you browse scenarios. Needs the agent server running:
          <code>uv run transcript_reader/agent_server.py</code> — or just open <code>http://localhost:8765</code>.
          Enter to send · Shift+Enter = newline.</div>
        <textarea id="aiq" placeholder="Ask about the scenario you're viewing…"></textarea>
      </div>
    </aside>
    <div class="modal" id="insmodal"><div class="modal-box">
      <div class="modal-h"><b id="insh"></b><button class="iconbtn" id="insclose">✕</button></div>
      <div class="modal-note">🌐 These rules are <b>global</b> — extracted once across <i>all</i> trajectories in this run and injected into <b>every</b> scenario. They are the <b>same for all scenarios</b> (not scenario-specific).</div>
      <div class="modal-body" id="insbody"></div>
    </div></div>
  </main>
</div>
<script>
const DATA = __DATA__;
const CATS = {
  too_easy:        {label:"Too easy",              color:"var(--too_easy)"},
  frontier_solved: {label:"Frontier · solved",     color:"var(--frontier_solved)"},
  frontier_unsolved:{label:"Frontier · improved, not solved", color:"var(--frontier_unsolved)"},
  beyond_frontier: {label:"Beyond frontier",       color:"var(--beyond_frontier)"},
};
const ORDER = ["too_easy","frontier_solved","frontier_unsolved","beyond_frontier"];
let RUN = Object.keys(DATA)[0];
let SEL = null;
let CATON = new Set(ORDER);
let HIDE = new Set();          // hidden attempt indices
let HDR_OPEN = true;           // scenario header collapsed state (persists across scenarios)
let CHAT = [];                 // Ask-AI conversation {role:'user'|'ai', content, pending?} — persists across scenarios
const AGENT_URL = "http://localhost:8765/ask";
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function scenarios(){ return DATA[RUN].scenarios; }
function scoreColor(v){ if(v==null) return "#999"; if(v>=7) return "#1d7a45"; if(v>=4) return "#c98a00"; return "#c0392b"; }
function spark(traj){
  const pts=traj.filter(v=>v!=null); if(!pts.length) return "";
  const w=54,h=16,max=10; const step=pts.length>1?w/(pts.length-1):0;
  const d=pts.map((v,i)=>`${i*step},${h-(v/max)*h}`).join(" ");
  return `<svg class="spark" width="${w}" height="${h}"><polyline points="${d}" fill="none" stroke="#555" stroke-width="1.5"/>`+
    pts.map((v,i)=>`<circle cx="${i*step}" cy="${h-(v/max)*h}" r="1.8" fill="${scoreColor(v)}"/>`).join("")+`</svg>`;
}

function renderRunToggle(){
  document.getElementById("runtoggle").innerHTML = Object.keys(DATA).map(r=>
    `<button class="${r===RUN?'active':''}" data-run="${esc(r)}">${esc(r)}</button>`).join("");
}
function renderCatFilter(){
  document.getElementById("catfilter").innerHTML = ORDER.map(c=>{
    const n=scenarios().filter(s=>s.category===c).length;
    return `<span class="catchip ${CATON.has(c)?'on':''}" data-cat="${c}"
      style="border-color:${CATS[c].color};color:${CATS[c].color}">${CATS[c].label} ${n}</span>`;
  }).join("");
}
function renderList(){
  const q=document.getElementById("search").value.toLowerCase();
  const box=document.getElementById("list"); box.innerHTML="";
  for(const c of ORDER){
    if(!CATON.has(c)) continue;
    const items=scenarios().filter(s=>s.category===c &&
      (!q || (s.title+" "+s.scenario).toLowerCase().includes(q)));
    if(!items.length) continue;
    box.insertAdjacentHTML("beforeend",`<div class="grp-h" style="color:${CATS[c].color}">${CATS[c].label} · ${items.length}</div>`);
    for(const s of items){
      box.insertAdjacentHTML("beforeend",
        `<div class="item ${SEL===s.id?'sel':''}" data-id="${esc(s.id)}">
           <span class="dot" style="background:${CATS[c].color}"></span>
           <span class="ttl">${esc(s.title)}</span>
           <span class="meta">${s.attempts.length}× ${spark(s.goal_traj)}</span>
         </div>`);
    }
  }
}
function turnHtml(t, learner){
  const isL = t.speaker===learner;
  const isAct = /^\s*\[/.test(t.content||"") || (t.content||"").length===0;
  const cls = isAct?"act":(isL?"learner":"partner");
  return `<div class="turn ${cls}"><div class="sp">${esc(t.speaker)}<span class="tn">turn ${esc(t.turn)}</span></div>${esc(t.content)}</div>`;
}
function attemptCol(a, idx, s){
  const g=a.scores.goal, r=a.scores.relationship;
  const otherChips=["believability","knowledge","social_rules","secret","financial_and_material_benefits"]
    .map(k=>a.scores[k]!=null?`<span class="chip">${k.slice(0,3)} <b>${a.scores[k]}</b></span>`:"").join("");
  return `<div class="col" data-att="${idx}">
    <div class="col-h">
      <div class="a-title">Attempt ${a.attempt!=null?a.attempt:idx+1}
        <span class="solvedtag ${a.solved?'solved':'unsolved'}">${a.solved?'✓ solved':'✗ not solved'}</span></div>
      <div class="chips">
        <span class="chip">GOAL <b style="color:${scoreColor(g)}">${g??'–'}</b></span>
        <span class="chip">REL <b style="color:${scoreColor(r)}">${r??'–'}</b></span>${otherChips}
      </div>
    </div>
    <div class="turns">${a.transcript.map(t=>turnHtml(t,s.learner)).join("")||'<div class="empty">no transcript</div>'}</div>
    ${a.reflexion?`<details class="reflex"><summary>🧠 Reflection used this attempt (from prior try)</summary><div class="reflex-body">${esc(a.reflexion)}</div></details>`:''}
  </div>`;
}
function renderScenario(){
  const s=scenarios().find(x=>x.id===SEL);
  const content=document.getElementById("content");
  if(!s){content.innerHTML='<div class="empty">Select a scenario.</div>';return;}
  const cat=CATS[s.category];
  document.getElementById("crumb").textContent = `${RUN} · ${s.learner} vs ${s.partner}`;
  // attempt show/hide controls
  document.getElementById("attctl").innerHTML = s.attempts.map((a,i)=>
    `<label><input type="checkbox" data-hide="${i}" ${HIDE.has(i)?'':'checked'}> ${i+1}</label>`).join("");
  const cols=s.attempts.map((a,i)=>HIDE.has(i)?'':attemptCol(a,i,s)).join("");
  content.innerHTML = `
    <details class="header" ${HDR_OPEN?'open':''} id="hdr">
      <summary>
        <span class="h-title">${esc(s.title)}</span>
        <span class="badge" style="background:${cat.color}">${cat.label}</span>
        <span class="goalrow">${spark(s.goal_traj)}</span>
        <span style="font-size:11.5px;color:var(--muted)">op: ${esc(s.mutation_operator||'—')} · depth ${esc(s.lineage_depth??'–')} · lp ${esc(s.lp_value??'–')}</span>
      </summary>
      <div class="hbody">
        <div class="kv"><b>Scenario:</b> ${esc(s.scenario)}</div>
        <div class="kv"><b>${esc(s.learner)} (learner):</b> ${esc(s.learner_goal)}</div>
        <div class="kv"><b>${esc(s.partner)} (partner):</b> ${esc(s.partner_goal)}</div>
        ${s.rubric?`<div class="kv"><b>Success rubric:</b> ${esc(s.rubric)}</div>`:''}
      </div>
    </details>
    <div class="cols">${cols||'<div class="empty">All attempts hidden.</div>'}</div>`;
  const hdr=document.getElementById("hdr");
  if(hdr) hdr.addEventListener("toggle",()=>{HDR_OPEN=hdr.open;});
}
function renderInsights(){   // run-level, shown in the modal — NOT tied to a scenario
  const ins=DATA[RUN].insights||[];
  document.getElementById("insh").textContent=`Global insights · ${RUN} · ${ins.length} rules`;
  document.getElementById("insbody").innerHTML = ins.length
    ? ins.map((r,i)=>`<div class="insight"><b>${i+1}.</b> ${esc(r)}</div>`).join("")
    : '<div class="insight">(none)</div>';
}
function selectScenario(id){ SEL=id; HIDE=new Set(); renderList(); renderScenario(); }

// ---- AI agent (talks to the local agent_server.py) ----
function buildContext(s){
  const lines=[];
  lines.push(`RUN: ${RUN}`);
  lines.push(`TITLE: ${s.title}`);
  lines.push(`CATEGORY: ${s.category}  (classification=${s.classification}, solved=${s.solved}, lp=${s.lp_value})`);
  lines.push(`SCENARIO: ${s.scenario}`);
  lines.push(`LEARNER (${s.learner}) GOAL: ${s.learner_goal}`);
  lines.push(`PARTNER (${s.partner}) GOAL: ${s.partner_goal}`);
  if(s.rubric) lines.push(`SUCCESS RUBRIC: ${s.rubric}`);
  s.attempts.forEach((a,i)=>{
    lines.push(`\n===== ATTEMPT ${a.attempt??i+1} — ${a.solved?'SOLVED':'NOT SOLVED'} (goal=${a.scores.goal}, rel=${a.scores.relationship}) =====`);
    if(a.reflexion) lines.push(`[reflection guiding this attempt]: ${a.reflexion}`);
    a.transcript.forEach(t=>lines.push(`[t${t.turn}] ${t.speaker}: ${t.content}`));
  });
  return lines.join("\n");
}
function renderChat(){
  const box=document.getElementById("chat");
  if(!CHAT.length){box.innerHTML='<div class="chat-empty">Ask about the scenario you\'re viewing.<br>The conversation stays as you move between scenarios.</div>';return;}
  box.innerHTML=CHAT.map(m=>`<div class="msg ${m.role}${m.pending?' pending':''}">${esc(m.content)}</div>`).join("");
  box.scrollTop=box.scrollHeight;
}
async function pingAgent(){
  const st=document.getElementById("aistatus");
  try{const r=await fetch(AGENT_URL.replace(/\/ask$/,"/health"),{cache:"no-store"});
    if(r.ok){st.textContent="🟢 connected";st.className="aistatus up";return true;}}catch(e){}
  st.textContent="🔴 server off";st.className="aistatus down";return false;
}
async function askAI(){
  const ta=document.getElementById("aiq"); const q=ta.value.trim(); if(!q)return;
  ta.value="";
  const s=scenarios().find(x=>x.id===SEL);
  CHAT.push({role:"user",content:q});
  CHAT.push({role:"ai",content:"…thinking",pending:true});
  renderChat();
  const history=CHAT.filter(m=>!m.pending).map(m=>({role:m.role==="ai"?"assistant":"user",content:m.content}));
  try{
    const r=await fetch(AGENT_URL,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({messages:history, context:s?buildContext(s):"", scenario_id:s?s.id:null, run:RUN})});
    const j=await r.json().catch(()=>({}));
    CHAT.pop();
    CHAT.push({role:"ai",content: r.ok?(j.answer||"(empty response)"):("⚠️ Server error "+r.status+": "+(j.error||""))});
  }catch(e){
    CHAT.pop();
    CHAT.push({role:"ai",content:"⚠️ Can't reach the agent server. Run  uv run transcript_reader/agent_server.py  in a terminal (or open http://localhost:8765), then send again."});
  }
  pingAgent(); renderChat();
}

// events
document.getElementById("runtoggle").onclick=e=>{const b=e.target.closest("button");if(!b)return;
  RUN=b.dataset.run;SEL=null;renderRunToggle();renderCatFilter();renderList();renderScenario();};
document.getElementById("catfilter").onclick=e=>{const c=e.target.closest(".catchip");if(!c)return;
  const k=c.dataset.cat; CATON.has(k)?CATON.delete(k):CATON.add(k); renderCatFilter(); renderList();};
document.getElementById("search").oninput=renderList;
document.getElementById("list").onclick=e=>{const it=e.target.closest(".item");if(it)selectScenario(it.dataset.id);};
document.getElementById("collapse").onclick=()=>document.getElementById("sidebar").classList.toggle("collapsed");
document.getElementById("drawerbtn").onclick=()=>{const d=document.getElementById("drawer");
  d.classList.toggle("open"); if(d.classList.contains("open")){pingAgent();renderChat();document.getElementById("aiq").focus();}};
document.getElementById("drawerclose").onclick=()=>document.getElementById("drawer").classList.remove("open");
document.getElementById("insbtn").onclick=()=>{renderInsights();document.getElementById("insmodal").classList.add("open");};
document.getElementById("insclose").onclick=()=>document.getElementById("insmodal").classList.remove("open");
document.getElementById("insmodal").onclick=e=>{if(e.target.id==="insmodal")e.currentTarget.classList.remove("open");};
document.getElementById("aiclear").onclick=()=>{CHAT=[];renderChat();};
document.getElementById("aiq").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();askAI();}});
document.getElementById("attctl").onchange=e=>{const cb=e.target.closest("[data-hide]");if(!cb)return;
  const i=+cb.dataset.hide; cb.checked?HIDE.delete(i):HIDE.add(i); renderScenario();};
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT")return;
  const cur=scenarios().filter(s=>CATON.has(s.category));
  const i=cur.findIndex(s=>s.id===SEL);
  if(e.key==="ArrowDown"&&i<cur.length-1)selectScenario(cur[i+1].id);
  if(e.key==="ArrowUp"&&i>0)selectScenario(cur[i-1].id);
});
// init
renderRunToggle();renderCatFilter();renderList();renderChat();pingAgent();
</script></body></html>"""


if __name__ == "__main__":
    main()
