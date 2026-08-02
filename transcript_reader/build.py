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
        # Hidden ground truth the partner is driven by (§3.3). Never shown to the learner;
        # shown here because `solved` conjoins key_check_passed, so an attempt with GOAL 9
        # can still read "not solved" and the key is the only way to see why.
        "partner_key": d.get("partner_key") or None,
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
/* partner key (hidden ground truth) */
.keybox{margin-top:8px;border:1px solid #d9c9a8;background:#fdf9f0;border-radius:8px;padding:8px 11px}
.keybox{padding:0}
.keybox>summary{cursor:pointer;list-style:none;padding:8px 11px;user-select:none}
.keybox>summary::-webkit-details-marker{display:none}
.keybox>summary::before{content:"▸ ";color:#8a6d3b}
.keybox[open]>summary::before{content:"▾ "}
.keybox[open]>summary{padding-bottom:0}
.keybox .kbody{padding:0 11px 9px}
.keybox .kh{display:inline;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#8a6d3b}
.keybox .khint{font-size:11px;color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0;margin-left:8px}
.keybox .kh .mech{background:#8a6d3b;color:#fff;padding:1px 7px;border-radius:9px;margin-left:6px;text-transform:none;letter-spacing:0}
.keybox h5{margin:7px 0 3px;font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
.keybox ol{margin:0;padding-left:20px}
.keybox li{font-size:12.5px;line-height:1.45;margin:2px 0}
.keybox li.trig{color:#a03a2c}
.keybox .kmisc{font-size:12.5px;line-height:1.45;margin:3px 0;color:#444}
/* per-attempt key-check verdict */
.kcheck{margin-top:6px;padding-top:6px;border-top:1px dashed var(--line);font-size:11.5px}
.kcheck .kcline{display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin-bottom:3px}
.kctag{font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:9px;cursor:help}
.kctag.met{background:#e4f5ec;color:#1d7a45} .kctag.unmet{background:#f0eeea;color:#8a8580}
.kctag.tripped{background:#fdecea;color:#c0392b} .kctag.repaired{background:#fff4dd;color:#b9821f}
.kcrat{color:var(--muted);line-height:1.4;font-style:italic}
/* review notes + reviewer checkmarks */
.who{display:flex;gap:3px;align-items:center;font-size:11px;color:var(--muted)}
.who button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:3px 7px;cursor:pointer;font-size:11px;font-weight:700}
.who button.me{background:var(--ink);color:#fff;border-color:var(--ink)}
.ckbtn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:5px 9px;cursor:pointer;font-size:12px;white-space:nowrap}
.ckbtn.on{background:#e4f5ec;border-color:#1d7a45;color:#1d7a45;font-weight:700}
.ckwho{font-size:10px;padding:1px 5px;border-radius:8px;background:#e4f5ec;color:#1d7a45;font-weight:700;margin-left:3px}
.ckwho.hj{background:#eef0fb;color:#3b4a9a}
.item .cks{display:flex;gap:2px;flex:0 0 auto}
.notes-area{width:100%;min-height:220px;padding:9px 11px;border:1px solid var(--line);border-radius:8px;
  font:inherit;font-size:13px;line-height:1.5;resize:vertical;box-sizing:border-box}
.notes-status{font-size:11px;color:var(--muted);margin-top:6px;min-height:14px}
.notes-meta{font-size:11.5px;color:var(--muted);line-height:1.5;margin:8px 0 0}
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
      <div class="who" id="who" title="Who is reviewing right now — stamps your checkmarks"></div>
      <button class="ckbtn" id="ckbtn" title="Mark this scenario reviewed by you">☐ Reviewed</button>
      <button class="iconbtn" id="notesbtn" title="Notes for this scenario (shared, committed to git)">📝 Notes</button>
      <button class="iconbtn" id="insbtn" title="Global insights for this run (same for all scenarios)">🌐 Insights</button>
    </div>
    <div id="content" style="flex:1;display:flex;flex-direction:column;overflow:hidden">
      <div class="empty">Select a scenario from the list.</div>
    </div>
    <aside class="drawer" id="notesdrawer">
      <div class="drawer-h"><b>📝 Review notes</b><span class="aistatus" id="notestatus">…</span>
        <span style="flex:1"></span>
        <button class="iconbtn" id="notesclose">✕</button></div>
      <div class="drawer-body">
        <div id="notesfor" style="font-size:12.5px;font-weight:700;margin-bottom:8px"></div>
        <textarea class="notes-area" id="notestext" placeholder="Notes on this scenario — what you checked, what looks wrong, what to follow up.&#10;&#10;Saved to transcript_reader/review_notes.json and committed, so both reviewers see them."></textarea>
        <div class="notes-status" id="notessaved"></div>
        <div class="notes-meta" id="notesmeta"></div>
        <div class="notes-meta">Autosaves ~1s after you stop typing, to
          <code>transcript_reader/review_notes.json</code>. Commit that file to share your review.
          With the server off, notes stay in this browser only and are NOT shared — the badge
          above turns red.</div>
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
let KEY_OPEN = true;           // partner-key block open state (persists across scenarios)
const NOTES_URL = "http://localhost:8765/notes";
const REVIEWERS = ["HX","HJ"];          // Huanxing / Huijun — stamped on checkmarks
let ME = localStorage.getItem("reader_me") || "HX";
let NOTES = {};                          // {id: {notes, checked:{HX:iso,HJ:iso}, title}}
let NOTES_OK = false;                    // false => server down, notes are browser-local only
let saveTimer = null;
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
           <span class="ttl">${esc(s.title)}${noteFor(s.id).notes?' <span title="has notes">📝</span>':''}</span>
           <span class="cks">${checkBadges(s.id)}</span>
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
// Per-attempt key-check verdict. `solved` = GOAL>=7 AND REL>=0 AND judge_goal_achieved
// AND key_check_passed — so this row is the only way to see why a high-GOAL attempt
// still reads "not solved". Indices in the verdict are 0-based into partner_key.
function keyCheckHtml(a, s){
  const kc=a.key_check; if(!kc) return "";
  const pk=s.partner_key||{};
  const conds=pk.movement_conditions||[], trigs=pk.hardening_triggers||[];
  const met=new Set(kc.conditions_met||[]), tripped=new Set(kc.triggers_tripped||[]),
        repaired=new Set(kc.triggers_repaired||[]);
  const tags=[];
  conds.forEach((c,i)=>tags.push(
    `<span class="kctag ${met.has(i)?'met':'unmet'}" title="${esc(c)}">C${i+1} ${met.has(i)?'✓ met':'· unmet'}</span>`));
  trigs.forEach((t,i)=>{
    if(!tripped.has(i)) return;
    const rep=repaired.has(i);
    tags.push(`<span class="kctag ${rep?'repaired':'tripped'}" title="${esc(t)}">T${i+1} ${rep?'⚠ tripped, repaired':'⚡ tripped'}</span>`);
  });
  const pass=kc.key_check_passed;
  return `<div class="kcheck">
    <div class="kcline">
      <span class="kctag ${pass?'met':'tripped'}">🔑 key check ${pass?'PASS':'FAIL'}</span>${tags.join("")}
    </div>
    ${kc.rationale?`<div class="kcrat">${esc(kc.rationale)}</div>`:''}
  </div>`;
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
      ${keyCheckHtml(a,s)}
    </div>
    <div class="turns">${a.transcript.map(t=>turnHtml(t,s.learner)).join("")||'<div class="empty">no transcript</div>'}</div>
    ${a.reflexion?`<details class="reflex"><summary>🧠 Reflection used this attempt (from prior try)</summary><div class="reflex-body">${esc(a.reflexion)}</div></details>`:''}
  </div>`;
}
// The partner's actual setup — hidden from BOTH agents at run time, but it defines what
// "solved" means, so it's the thing you need in front of you while reading a transcript.
function partnerKeyHtml(s){
  const k=s.partner_key;
  if(!k) return `<div class="kv" style="color:var(--muted)"><i>No partner_key (unkeyed scenario — key check auto-passes).</i></div>`;
  const conds=k.movement_conditions||[], trigs=k.hardening_triggers||[];
  return `<details class="keybox" id="keybox" ${KEY_OPEN?'open':''}>
    <summary><span class="kh">🔒 Partner key · hidden ground truth</span>${k.key_mechanism?`<span class="mech badge">${esc(k.key_mechanism)}</span>`:''}<span class="khint">${conds.length} conditions · ${trigs.length} triggers</span></summary>
    <div class="kbody">
      <h5>Movement conditions — the learner must genuinely satisfy ≥1</h5>
      <ol>${conds.map(c=>`<li>${esc(c)}</li>`).join("")||'<li><i>none</i></li>'}</ol>
      <h5>Hardening triggers — tripping one un-repaired fails the key check</h5>
      <ol>${trigs.map(t=>`<li class="trig">${esc(t)}</li>`).join("")||'<li><i>none</i></li>'}</ol>
      ${k.surface_misdirection?`<div class="kmisc"><b>Surface misdirection</b> <span style="color:var(--muted)">— what the partner SAYS the problem is; injected into the partner's own prompt as their cover story. The only key field allowed to appear publicly.</span><br>${esc(k.surface_misdirection)}</div>`:''}
      ${k.cost_coupling?`<div class="kmisc"><b>Cost coupling</b> <span style="color:var(--muted)">— what meeting the conditions costs the LEARNER's own goal. Design-time only: never injected into any agent prompt, never checked at scoring time.</span><br>${esc(k.cost_coupling)}</div>`:''}
    </div>
  </details>`;
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
        ${partnerKeyHtml(s)}
        ${s.rubric?`<div class="kv"><b>Success rubric:</b> ${esc(s.rubric)}</div>`:''}
      </div>
    </details>
    <div class="cols">${cols||'<div class="empty">All attempts hidden.</div>'}</div>`;
  const hdr=document.getElementById("hdr");
  if(hdr) hdr.addEventListener("toggle",()=>{HDR_OPEN=hdr.open;});
  const kb=document.getElementById("keybox");
  if(kb) kb.addEventListener("toggle",e=>{e.stopPropagation();KEY_OPEN=kb.open;});
}
function renderInsights(){   // run-level, shown in the modal — NOT tied to a scenario
  const ins=DATA[RUN].insights||[];
  document.getElementById("insh").textContent=`Global insights · ${RUN} · ${ins.length} rules`;
  document.getElementById("insbody").innerHTML = ins.length
    ? ins.map((r,i)=>`<div class="insight"><b>${i+1}.</b> ${esc(r)}</div>`).join("")
    : '<div class="insight">(none)</div>';
}
function selectScenario(id){ SEL=id; HIDE=new Set(); renderList(); renderScenario(); renderNotes(); }

// ---- review notes + checkmarks (shared via the agent server; committed to git) ----
// Falls back to localStorage when the server is off so you never silently lose typing,
// but that copy is NOT shared — the status badge says so.
function noteFor(id){ return NOTES[id] || {notes:"", checked:{}}; }
function localCache(){ try{return JSON.parse(localStorage.getItem("reader_notes")||"{}");}catch(e){return {};} }
async function loadNotes(){
  const st=document.getElementById("notestatus");
  try{
    const r=await fetch(NOTES_URL,{cache:"no-store"});
    if(!r.ok) throw new Error(r.status);
    NOTES=await r.json(); NOTES_OK=true;
    st.textContent="🟢 shared"; st.className="aistatus up";
  }catch(e){
    NOTES=localCache(); NOTES_OK=false;
    st.textContent="🔴 local only"; st.className="aistatus down";
  }
  renderList(); renderNotes();
}
async function saveNote(id, patch){
  const e=noteFor(id);
  if(patch.notes!==undefined) e.notes=patch.notes;
  if(patch.checked!==undefined){
    e.checked=e.checked||{};
    for(const [w,v] of Object.entries(patch.checked)){ if(v) e.checked[w]=v; else delete e.checked[w]; }
  }
  NOTES[id]=e;
  const s=document.getElementById("notessaved");
  const title=(scenarios().find(x=>x.id===id)||{}).title||"";
  try{
    const r=await fetch(NOTES_URL,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({id, title, notes:e.notes, checked:patch.checked||{}})});
    if(!r.ok) throw new Error(r.status);
    NOTES_OK=true;
    if(s) s.textContent="✓ saved to review_notes.json — commit it to share";
    document.getElementById("notestatus").textContent="🟢 shared";
    document.getElementById("notestatus").className="aistatus up";
  }catch(err){
    NOTES_OK=false;
    const c=localCache(); c[id]=e; localStorage.setItem("reader_notes",JSON.stringify(c));
    if(s) s.textContent="⚠️ server off — saved in this browser only, NOT shared";
    document.getElementById("notestatus").textContent="🔴 local only";
    document.getElementById("notestatus").className="aistatus down";
  }
  renderList(); renderCheckBtn();
}
function renderWho(){
  document.getElementById("who").innerHTML = "as " + REVIEWERS.map(w=>
    `<button class="${w===ME?'me':''}" data-who="${w}">${w}</button>`).join("");
}
function checkBadges(id){
  const c=noteFor(id).checked||{};
  return REVIEWERS.filter(w=>c[w]).map(w=>
    `<span class="ckwho ${w.toLowerCase()}" title="checked by ${w} on ${esc(c[w])}">✓${w}</span>`).join("");
}
function renderCheckBtn(){
  const b=document.getElementById("ckbtn"); if(!b) return;
  if(!SEL){ b.textContent="☐ Reviewed"; b.className="ckbtn"; return; }
  const c=noteFor(SEL).checked||{};
  const mine=!!c[ME];
  b.className="ckbtn"+(mine?" on":"");
  b.textContent=(mine?"☑":"☐")+` Reviewed by ${ME}`;
  b.title=mine?`Checked ${c[ME]} — click to un-check`:`Mark reviewed as ${ME}`;
}
function renderNotes(){
  const box=document.getElementById("notestext"); if(!box) return;
  const s=scenarios().find(x=>x.id===SEL);
  document.getElementById("notesfor").textContent = s?s.title:"(no scenario selected)";
  box.value = s?noteFor(s.id).notes||"":"";
  box.disabled = !s;
  document.getElementById("notessaved").textContent="";
  const c=s?(noteFor(s.id).checked||{}):{};
  document.getElementById("notesmeta").innerHTML = REVIEWERS.map(w=>
    `${w}: ${c[w]?"✓ "+esc(new Date(c[w]).toLocaleString()):"—"}`).join(" · ");
  renderCheckBtn();
}

// events
document.getElementById("runtoggle").onclick=e=>{const b=e.target.closest("button");if(!b)return;
  RUN=b.dataset.run;SEL=null;renderRunToggle();renderCatFilter();renderList();renderScenario();};
document.getElementById("catfilter").onclick=e=>{const c=e.target.closest(".catchip");if(!c)return;
  const k=c.dataset.cat; CATON.has(k)?CATON.delete(k):CATON.add(k); renderCatFilter(); renderList();};
document.getElementById("search").oninput=renderList;
document.getElementById("list").onclick=e=>{const it=e.target.closest(".item");if(it)selectScenario(it.dataset.id);};
function toggleSidebar(force){
  const sb=document.getElementById("sidebar");
  const hidden = force===undefined ? !sb.classList.contains("collapsed") : force;
  sb.classList.toggle("collapsed", hidden);
  localStorage.setItem("reader_sidebar_hidden", hidden?"1":"0");
  const b=document.getElementById("collapse");
  b.textContent = hidden ? "▶" : "☰";
  b.title = (hidden?"Show":"Hide")+" scenario list  (press \\ )";
}
document.getElementById("collapse").onclick=()=>toggleSidebar();
document.getElementById("insbtn").onclick=()=>{renderInsights();document.getElementById("insmodal").classList.add("open");};
document.getElementById("insclose").onclick=()=>document.getElementById("insmodal").classList.remove("open");
document.getElementById("insmodal").onclick=e=>{if(e.target.id==="insmodal")e.currentTarget.classList.remove("open");};
document.getElementById("attctl").onchange=e=>{const cb=e.target.closest("[data-hide]");if(!cb)return;
  const i=+cb.dataset.hide; cb.checked?HIDE.delete(i):HIDE.add(i); renderScenario();};
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA")return;  // don't steal keys while typing notes
  if(e.key==="\\"){ toggleSidebar(); return; }          // full-width transcript reading
  if(e.key==="Escape"){ document.getElementById("notesdrawer").classList.remove("open"); return; }
  const cur=scenarios().filter(s=>CATON.has(s.category));
  const i=cur.findIndex(s=>s.id===SEL);
  if(e.key==="ArrowDown"&&i<cur.length-1)selectScenario(cur[i+1].id);
  if(e.key==="ArrowUp"&&i>0)selectScenario(cur[i-1].id);
});
document.getElementById("notesbtn").onclick=()=>{const d=document.getElementById("notesdrawer");
  d.classList.toggle("open"); if(d.classList.contains("open")){renderNotes();document.getElementById("notestext").focus();}};
document.getElementById("notesclose").onclick=()=>document.getElementById("notesdrawer").classList.remove("open");
document.getElementById("who").onclick=e=>{const b=e.target.closest("button");if(!b)return;
  ME=b.dataset.who; localStorage.setItem("reader_me",ME); renderWho(); renderCheckBtn(); renderNotes();};
document.getElementById("ckbtn").onclick=()=>{
  if(!SEL) return;
  const c=noteFor(SEL).checked||{};
  saveNote(SEL,{checked:{[ME]: c[ME]?null:new Date().toISOString()}});
};
document.getElementById("notestext").addEventListener("input",e=>{
  if(!SEL) return;
  const v=e.target.value;
  document.getElementById("notessaved").textContent="saving…";
  clearTimeout(saveTimer);
  saveTimer=setTimeout(()=>saveNote(SEL,{notes:v}),900);   // debounce: one write per pause
});
// init
renderRunToggle();renderCatFilter();renderWho();renderList();loadNotes();
toggleSidebar(localStorage.getItem("reader_sidebar_hidden")==="1");
</script></body></html>"""


if __name__ == "__main__":
    main()
