#!/usr/bin/env python3
"""Build a single-scenario showcase reader: partner_key v1 vs v2, side by side.

The full reader (`build.py`) is a browser over ~90 scenarios. This is the opposite: ONE pair of
scenarios, rendered to make the schema change legible to someone who has never seen the project.

The default pair is chosen so the comparison is apples-to-apples: both children descend from the SAME
raw SOTOPIA seed (`01H7VFHNGJEVGSVPPT0784H6P8`), so the only differences are the schema the generator
wrote against and the pipeline that admitted them.

    old  gen-90         v1 key: movement_conditions + surface_misdirection + cost_coupling
    new  matrix_v1      v2 key: movement_conditions demoted to witness, internal_state graded

The new scenario also happens to be the run's cleanest demonstration that the items discriminate:
gpt-5-mini reaches the state on attempt 3 (goal 10), while gpt-4o-mini trips BOTH hardening triggers
and never gets above goal 2. Section 3 renders that from the crossplay record.

    uv run transcript_reader/build_showcase.py
    open transcript_reader/showcase.html

Every path is overridable, so this generalises to any (old, new) pair:

    uv run transcript_reader/build_showcase.py \
        --old results/gen90_expel/bank/generated/<id>.json \
        --new results/matrix_v1/sets/gpt5mini/bank/generated/<id>.json \
        --crossplay results/matrix_v1/crossplay/gpt5mini__gpt4omini/episodes/<id>.json:gpt-4o-mini
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "showcase.html"

DEFAULT_OLD = "results/gen90_expel/bank/generated/a49fcb6b-1ddd-498d-8af2-bc13258bf1ae_p0.json"
DEFAULT_NEW = ("results/matrix_v1/sets/gpt5mini/bank/generated/"
               "01H7VFHNGJEVGSVPPT0784H6P8__gpt5mini_p0.json")
DEFAULT_XPLAY = ("results/matrix_v1/crossplay/gpt5mini__gpt4omini/episodes/"
                 "01H7VFHNGJEVGSVPPT0784H6P8__gpt5mini_p0.json:gpt-4o-mini")

DIMS = ["goal", "relationship", "believability", "knowledge",
        "secret", "social_rules", "financial_and_material_benefits"]

# Field → (display label, role tag, prose explaining the field's JOB in the design).
# The role tag is the whole point of the showcase: in v1 nothing said which field was the
# rubric, so movement_conditions became both the proof-of-solvability and the answer key.
FIELD_ROLES = {
    "key_mechanism": (
        "key_mechanism", "label",
        "A short psychology label. In v1 this was a closed enum of five tags; in v2 it is free "
        "text, because an enum caps how many kinds of person the generator can invent."),
    "internal_state": (
        "internal_state", "graded",
        "NEW IN v2. What is actually true of this person, written as a fact about them rather than "
        "a requirement on the learner. This is the graded object: any route that reaches it counts."),
    "movement_conditions": (
        "movement_conditions", "witness",
        "Concrete behaviours that would move them. In v2 this is a WITNESS — the oracle's proof that "
        "a path exists — and no longer the rubric. In v1 it was both, which made it an answer key."),
    "hardening_triggers": (
        "hardening_triggers", "denies",
        "What makes them close up. In v2 these must actively deny the internal_state, so the "
        "scenario has an internal logic instead of a list of dos and don'ts."),
    "surface_misdirection": (
        "surface_misdirection", "retired",
        "RETIRED IN v2. The stated-versus-actual gap. In v1 this was the only field holding the "
        "real psychology — while being explicitly labelled a decoy — and it was written from a "
        "confused point of view, mixing what the learner hears with what the partner believes."),
    "cost_coupling": (
        "cost_coupling", "retired",
        "RETIRED IN v2. What satisfying the partner costs the learner. Never gated on, and it "
        "pushed the generator toward transactional trade-off scenarios."),
    "version": ("version", "meta", "Schema version stamp."),
}

ROLE_ORDER = ["graded", "witness", "denies", "label", "retired", "meta"]


def agent_name(p) -> str:
    if not isinstance(p, dict):
        return str(p)
    return ((p.get("first_name") or "") + " " + (p.get("last_name") or "")).strip() or "Agent"


def _people(d: dict):
    profiles = d.get("agent_profiles") or []
    tidx = d.get("target_agent_idx", 0) or 0
    learner = agent_name(profiles[tidx]) if tidx < len(profiles) else "Learner"
    partner = agent_name(profiles[1 - tidx]) if len(profiles) > 1 else "Partner"
    goals = d.get("agent_goals") or []
    return {
        "learner": learner,
        "partner": partner,
        "learner_goal": goals[tidx] if tidx < len(goals) else "",
        "partner_goal": goals[1 - tidx] if len(goals) > 1 else "",
    }


def key_fields(d: dict) -> list:
    """partner_key as ordered, role-tagged, explained field cards."""
    k = d.get("partner_key") or {}
    out = []
    for name, val in k.items():
        if val in (None, [], ""):
            continue
        if name == "version":
            continue        # bookkeeping, not mechanism — and see the caveat printed by main()
        label, role, why = FIELD_ROLES.get(name, (name, "meta", ""))
        out.append({"name": label, "role": role, "why": why,
                    "value": val if isinstance(val, list) else [str(val)],
                    "is_list": isinstance(val, list)})
    out.sort(key=lambda f: ROLE_ORDER.index(f["role"]) if f["role"] in ROLE_ORDER else 99)
    return out


def extract(path: Path, label: str, schema: str) -> dict:
    d = json.loads(path.read_text())
    attempts = []
    for a in d.get("attempts") or []:
        sc = a.get("scores") or {}
        attempts.append({
            "attempt": a.get("attempt"),
            "solved": bool(a.get("solved")),
            "scores": {k: sc.get(k) for k in DIMS},
            "verdict": a.get("key_check_result"),
            "reflexion": a.get("reflexion") or "",
            "transcript": [{"turn": t.get("turn"), "speaker": t.get("speaker", "?"),
                            "content": t.get("content", "")}
                           for t in (a.get("transcript") or [])],
        })
    rec = {
        "label": label, "schema": schema, "id": d.get("id"),
        "title": d.get("scenario_title") or "",
        "scenario": d.get("scenario") or "",
        "classification": d.get("classification") or "unknown",
        "solved": bool(d.get("terminal_success")),
        "operator": d.get("mutation_operator"),
        "root_seed": d.get("root_seed_env_pk"),
        "goal_traj": [g for g in (d.get("goal_trajectory") or []) if g is not None],
        "key": key_fields(d),
        "raw_key": d.get("partner_key") or {},
        "attempts": attempts,
    }
    rec.update(_people(d))
    return rec


def extract_crossplay(path: Path, learner_label: str, key_src: dict) -> dict:
    """Crossplay records store transcripts + verdicts flat, not as attempt objects."""
    d = json.loads(path.read_text())
    tx = d.get("transcripts") or []
    vd = d.get("key_check_verdicts") or []
    traj = [g for g in (d.get("goal_trajectory") or []) if g is not None]
    attempts = []
    for i, t in enumerate(tx):
        v = vd[i] if i < len(vd) else None
        attempts.append({
            "attempt": i + 1,
            "solved": bool((v or {}).get("key_check_passed")) and (
                traj[i] >= 7.0 if i < len(traj) else False),
            # crossplay keeps only terminal 7-dim scores; per-attempt we have goal alone
            "scores": {"goal": traj[i] if i < len(traj) else None},
            "verdict": v,
            "reflexion": "",
            "transcript": [{"turn": x.get("turn"), "speaker": x.get("speaker", "?"),
                            "content": x.get("content", "")} for x in (t or [])],
        })
    return {
        "label": learner_label, "schema": "v2", "id": d.get("id") or d.get("scenario_id"),
        "title": "", "scenario": "",
        "classification": d.get("band") or d.get("terminal_state") or "unknown",
        "solved": bool(d.get("solved")),
        "operator": None, "root_seed": d.get("root_seed_env_pk"),
        "goal_traj": traj,
        "key": key_fields({"partner_key": key_src}), "raw_key": key_src,
        "final_scores": d.get("final_scores") or {},
        "attempts": attempts,
        "learner": "", "partner": "", "learner_goal": "", "partner_goal": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=DEFAULT_OLD, help="gen-90 (v1 schema) scenario JSON")
    ap.add_argument("--new", default=DEFAULT_NEW, help="matrix (v2 schema) scenario JSON")
    ap.add_argument("--crossplay", default=DEFAULT_XPLAY, metavar="PATH:LABEL",
                    help="the SAME new scenario played by a different learner, as path:label. "
                         "Repeatable. Omit with --no-crossplay.")
    ap.add_argument("--no-crossplay", action="store_true")
    ap.add_argument("--new-learner-label", default="gpt-5-mini",
                    help="label for the learner that played the new scenario natively")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        f = Path(p)
        if not f.is_absolute():
            f = ROOT / f
        if not f.exists():                      # tolerate a glob, so ids can be abbreviated
            hits = sorted(glob.glob(str(f)))
            if not hits:
                print(f"not found: {p}", file=sys.stderr)
                sys.exit(2)
            f = Path(hits[0])
        return f

    old = extract(resolve(args.old), "gen-90", "v1")
    new = extract(resolve(args.new), args.new_learner_label, "v2")

    others = []
    if not args.no_crossplay and args.crossplay:
        spec = args.crossplay
        path, _, lab = spec.rpartition(":")
        if not path:                            # no label given
            path, lab = spec, "other learner"
        others.append(extract_crossplay(resolve(path), lab, new["raw_key"]))

    same_root = old.get("root_seed") and old["root_seed"] == new.get("root_seed")
    data = {
        "old": old, "new": new, "others": others,
        "same_root": bool(same_root),
        "root_seed": new.get("root_seed") or old.get("root_seed"),
    }

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")

    print(f"old  {old['id']}  [{old['classification']}] solved={old['solved']} goals={old['goal_traj']}")
    print(f"new  {new['id']}  [{new['classification']}] solved={new['solved']} goals={new['goal_traj']}")
    for o in others:
        print(f"     vs {o['label']:14} [{o['classification']}] solved={o['solved']} goals={o['goal_traj']}")
    print(f"same root seed: {same_root}  ({data['root_seed']})")
    print(f"\nWrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"Open it: open {out}")


# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>partner_key v1 → v2 — single-scenario showcase</title>
<style>
:root{
  --bg:#faf9f7; --panel:#fff; --ink:#1e1c1a; --muted:#8a8580; --line:#e6e2dc;
  --learner:#2563a8; --learner-bg:#eaf2fb; --partner:#8a6d3b; --partner-bg:#f6f0e6;
  --too_easy:#9aa0a6; --frontier:#2e8b57; --beyond_frontier:#c0392b; --unknown:#9aa0a6;
  --graded:#2e7d5b; --graded-bg:#e6f4ee;
  --witness:#2563a8; --witness-bg:#eaf2fb;
  --denies:#b25a00; --denies-bg:#fdf0e2;
  --retired:#a33; --retired-bg:#faecec;
  --label:#6b6b6b; --label-bg:#f1efec;
  --meta:#9aa0a6; --meta-bg:#f4f3f1;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:22px 20px 80px}
h1{font-size:21px;margin:0 0 4px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h2 .n{color:var(--ink);margin-right:8px}
.sub{color:var(--muted);font-size:13px;margin:0 0 6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:900px){.two{grid-template-columns:1fr}}
.badge{display:inline-block;font-size:11px;padding:2px 9px;border-radius:10px;color:#fff;font-weight:600}
.pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;border:1px solid var(--line);
  background:#fff;color:#555;margin-right:5px}
.tag{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
  padding:2px 7px;border-radius:5px}
.col-h{display:flex;align-items:center;gap:9px;margin-bottom:3px}
.col-h .t{font-weight:700;font-size:15px}
.schema-v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--muted)}
.field{border:1px solid var(--line);border-radius:8px;margin:9px 0;overflow:hidden}
.field.retired{opacity:.72;background:#fcfafa}
.f-h{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#fbfaf9;border-bottom:1px solid var(--line)}
.f-h code{font-size:12.5px;font-weight:700}
.f-why{padding:7px 10px;font-size:12px;color:#5a5550;background:#fdfdfc;border-bottom:1px dashed var(--line)}
.f-v{padding:8px 10px;font-size:13px}
.f-v ul{margin:0;padding-left:19px}
.f-v li{margin:4px 0}
.q{font-size:13.5px;line-height:1.55}
.kv{font-size:13px;margin:7px 0}
.kv b{display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:2px}
.spark{display:inline-flex;gap:3px;align-items:flex-end;height:34px;vertical-align:bottom}
.spark i{width:15px;background:var(--learner);border-radius:2px 2px 0 0;position:relative}
.spark i.win{background:var(--frontier)}
.spark i.zero{background:var(--line)}
.traj{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px}
.tabs button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:6px 11px;cursor:pointer;font-size:12.5px}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.tsplit{display:grid;grid-template-columns:1fr 320px;gap:14px;align-items:start}
@media(max-width:1000px){.tsplit{grid-template-columns:1fr}}
.turn{margin:6px 0;padding:7px 10px;border-radius:8px;font-size:13px}
.turn.learner{background:var(--learner-bg);border-left:3px solid var(--learner)}
.turn.partner{background:var(--partner-bg);border-left:3px solid var(--partner)}
.turn.act{background:transparent;border-left:3px solid var(--line);font-style:italic;color:var(--muted)}
.turn .sp{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}
.turn.learner .sp{color:var(--learner)} .turn.partner .sp{color:var(--partner)}
.turn .sp .tn{color:var(--muted);font-weight:500;margin-left:6px}
.rail .box{border:1px solid var(--line);border-radius:8px;padding:9px 11px;margin-bottom:9px;background:#fff}
.rail h4{margin:0 0 6px;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.hit{font-size:12.5px;margin:5px 0;padding-left:17px;position:relative}
.hit::before{position:absolute;left:0;top:0;font-weight:800}
.hit.yes::before{content:"✓";color:var(--frontier)} .hit.no::before{content:"✗";color:var(--beyond_frontier)}
.hit.trip::before{content:"⚑";color:var(--beyond_frontier)}
.ev{font-size:12px;color:#5a5550;background:#fbfaf9;border-left:2px solid var(--line);padding:4px 8px;margin:5px 0}
.sc{display:flex;flex-wrap:wrap;gap:5px}
.sc span{font-size:11px;padding:2px 7px;border-radius:9px;background:#f4f3f1;color:#555;font-variant-numeric:tabular-nums}
.sc span.g{background:var(--graded-bg);color:var(--graded);font-weight:700}
.sc span.neg{background:var(--retired-bg);color:var(--retired);font-weight:700}
.note{background:#fff8e1;border:1px solid #e9c46a;border-radius:8px;padding:10px 12px;font-size:12.5px;margin:12px 0}
.note b{color:#8a6d1f}
details.raw{margin-top:12px}
details.raw>summary{cursor:pointer;font-size:12px;color:var(--muted)}
pre{background:#fbfaf9;border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font-size:11.5px}
</style></head><body><div class="wrap" id="app"></div>
<script>
const D = __DATA__;
const esc = s => String(s==null?"":s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const clsColor = c => `var(--${['too_easy','frontier','beyond_frontier'].includes(c)?c:'unknown'})`;

function badge(rec){
  return `<span class="badge" style="background:${clsColor(rec.classification)}">${esc(rec.classification)}</span>`;
}
function spark(traj){
  if(!traj||!traj.length) return '';
  const bars = traj.map(g=>{
    const h = Math.max(2, Math.round((g/10)*34));
    const k = g>=7?'win':(g<=0?'zero':'');
    return `<i class="${k}" style="height:${h}px" title="goal ${g}"></i>`;
  }).join('');
  return `<span class="spark">${bars}</span> <span class="traj">[${traj.join(', ')}]</span>`;
}
function fieldCard(f){
  return `<div class="field ${f.role==='retired'?'retired':''}">
    <div class="f-h"><code>${esc(f.name)}</code>
      <span class="tag" style="background:var(--${f.role}-bg);color:var(--${f.role})">${esc(f.role)}</span></div>
    ${f.why?`<div class="f-why">${esc(f.why)}</div>`:''}
    <div class="f-v">${f.is_list
      ? `<ul>${f.value.map(v=>`<li>${esc(v)}</li>`).join('')}</ul>`
      : `<div class="q">${esc(f.value[0])}</div>`}</div></div>`;
}
function schemaCol(rec, title){
  return `<div class="card"><div class="col-h"><span class="t">${esc(title)}</span>
      <span class="schema-v">partner_key ${esc(rec.schema)}</span>${badge(rec)}</div>
    <div class="sub">${esc(rec.id)}${rec.operator?` · operator <b>${esc(rec.operator)}</b>`:''}
      · solved <b>${rec.solved?'yes':'no'}</b></div>
    ${rec.key.map(fieldCard).join('')}</div>`;
}

// ---- turns + verdict rail ----------------------------------------------------
function isAct(c){ return /left the conversation|\[(?:non-verbal|action)/i.test(c); }
function turns(att, learner){
  return (att.transcript||[]).map(t=>{
    const cls = isAct(t.content) ? 'act' : (t.speaker===learner ? 'learner' : 'partner');
    return `<div class="turn ${cls}"><div class="sp">${esc(t.speaker)}<span class="tn">turn ${t.turn}</span></div>
      ${esc(t.content)}</div>`;
  }).join('');
}
function scoreChips(sc){
  if(!sc) return '';
  return `<div class="sc">${Object.entries(sc).filter(([,v])=>v!=null).map(([k,v])=>{
    const c = k==='goal' ? 'g' : (v<0 ? 'neg' : '');
    return `<span class="${c}">${esc(k.slice(0,4))} ${v}</span>`;}).join('')}</div>`;
}
function rail(att, key){
  const v = att.verdict;
  if(!v) return `<div class="rail"><div class="box"><h4>grading</h4>
    <div class="sub">No key_check record for this attempt.</div></div></div>`;
  const mc = key.movement_conditions||[], ht = key.hardening_triggers||[];
  const met = (v.conditions_met||[]).map(i=>mc[i]).filter(Boolean);
  const trip = (v.triggers_tripped||[]).map(i=>ht[i]).filter(Boolean);
  const yn = (b,t)=>`<div class="hit ${b?'yes':'no'}">${esc(t)}</div>`;
  return `<div class="rail">
    <div class="box"><h4>scores</h4>${scoreChips(att.scores)}</div>
    <div class="box"><h4>internal_state — the graded object</h4>
      ${yn(v.internal_state_reached,'state reached')}
      ${v.internal_state_evidence?`<div class="ev">${esc(v.internal_state_evidence)}</div>`:''}
      ${yn(v.apprehended,'learner apprehended it')}
      ${yn(v.acted_consistently,'acted consistently with it')}</div>
    <div class="box"><h4>movement_conditions — witness, ${met.length}/${mc.length} met</h4>
      ${met.length?met.map(m=>`<div class="hit yes">${esc(m)}</div>`).join('')
                  :'<div class="sub">none met</div>'}</div>
    <div class="box"><h4>hardening_triggers — ${trip.length} tripped</h4>
      ${trip.length?trip.map(t=>`<div class="hit trip">${esc(t)}</div>`).join('')
                   :'<div class="sub">none tripped</div>'}</div>
    ${v.rationale?`<div class="box"><h4>judge rationale</h4><div class="q">${esc(v.rationale)}</div></div>`:''}
  </div>`;
}

// ---- transcript viewer state -------------------------------------------------
const runs = [D.new, ...D.others, D.old].filter(Boolean);
// Open on the attempt that best shows the mechanism: the first one that reached the state,
// else the highest-scoring. Opening on attempt 1 of a 3-attempt solve shows only the failure.
// Rank on (state reached, goal) so a 3-attempt solve opens on the solve, not on the first
// attempt that merely reached the state with a low goal score.
function bestAtt(r){
  const a = r.attempts||[];
  let bi = 0, bk = [-1,-1];
  a.forEach((x,i)=>{
    const k = [(x.verdict||{}).internal_state_reached?1:0, (x.scores||{}).goal ?? -1];
    if(k[0]>bk[0] || (k[0]===bk[0] && k[1]>bk[1])){ bk=k; bi=i; }
  });
  return bi;
}
let cur = 0, curAtt = bestAtt(runs[0]);
function viewer(){
  const r = runs[cur];
  const atts = r.attempts||[];
  if(curAtt >= atts.length) curAtt = 0;
  const a = atts[curAtt];
  const runTabs = runs.map((x,i)=>{
    const which = x===D.old ? `gen-90 (v1)` : `${x.label} (v2)`;
    return `<button class="${i===cur?'on':''}" onclick="pickRun(${i})">${esc(which)}</button>`;}).join('');
  const attTabs = atts.map((x,i)=>{
    const g = (x.scores||{}).goal;
    return `<button class="${i===curAtt?'on':''}" onclick="pickAtt(${i})">attempt ${x.attempt}${
      g!=null?` · goal ${g}`:''}</button>`;}).join('');
  const learner = r.learner || D.new.learner;
  return `<div class="tabs">${runTabs}</div><div class="tabs">${attTabs}</div>
    ${a?`<div class="tsplit"><div class="card">${turns(a, learner)}
      ${a.reflexion?`<details class="raw"><summary>reflection written after this attempt</summary>
        <div class="q" style="margin-top:6px">${esc(a.reflexion)}</div></details>`:''}</div>
      ${rail(a, r.raw_key||{})}</div>`:'<div class="card sub">No transcript.</div>'}`;
}
function pickRun(i){ cur=i; curAtt=bestAtt(runs[i]); render(); }
function pickAtt(i){ curAtt=i; render(); }

// ---- discrimination panel ----------------------------------------------------
function discrim(){
  const rows = [D.new, ...D.others].map(r=>`<div class="card">
    <div class="col-h"><span class="t">${esc(r.label)}</span>${badge(r)}</div>
    <div class="kv"><b>goal per attempt</b>${spark(r.goal_traj)}</div>
    <div class="kv"><b>outcome</b>${r.solved?'reached the internal state':'never reached it'}</div>
    ${(()=>{const v=(r.attempts||[]).map(a=>a.verdict).filter(Boolean);
      const trip=v.reduce((n,x)=>n+((x.triggers_tripped||[]).length),0);
      const met=v.reduce((n,x)=>n+((x.conditions_met||[]).length),0);
      const st=v.filter(x=>x.internal_state_reached).length;
      return `<div class="kv"><b>internal state reached</b>
        <span style="font-size:15px;font-weight:700;color:${st?'var(--graded)':'var(--beyond_frontier)'}">
          ${st} of ${v.length} attempts</span></div>
        <div class="kv"><b>across all attempts</b>
        ${met} movement condition${met===1?'':'s'} met ·
        <span style="color:${trip>met?'var(--beyond_frontier)':'inherit'}">${trip} hardening trigger${
          trip===1?'':'s'} tripped</span></div>`;})()}
  </div>`).join('');
  return `<div class="two">${rows}</div>`;
}

function render(){
  const o=D.old, n=D.new;
  document.getElementById('app').innerHTML = `
  <h1>partner_key v1 → v2, on one scenario</h1>
  <p class="sub">${D.same_root
      ? `Both scenarios are children of the same raw SOTOPIA seed <code>${esc(D.root_seed)}</code>, so the
         schema and the pipeline are the only things that differ.`
      : `Root seeds differ — this is a schema illustration, not a controlled comparison.`}</p>

  <h2><span class="n">1</span>The schema change</h2>
  <p class="sub">Role tags are the point. In v1 nothing marked which field was the rubric, so
     <code>movement_conditions</code> served as both the proof that the scenario was solvable and the
     list of accepted answers.</p>
  <div class="two">${schemaCol(o,'Old — gen-90')}${schemaCol(n,'New — matrix_v1')}</div>
  <div class="note"><b>Read the two <code>movement_conditions</code> blocks against each other.</b>
     In v1 they are a checklist of active-listening moves, and the real psychology sits in
     <code>surface_misdirection</code> — a field explicitly labelled a decoy. In v2 the psychology moves
     into <code>internal_state</code> and becomes the graded object, the conditions demote to a witness,
     and the triggers are written to deny the state rather than to enumerate mistakes.</div>

  <h2><span class="n">2</span>The new scenario</h2>
  <div class="card">
    <div class="kv"><b>context</b><div class="q">${esc(n.scenario)}</div></div>
    <div class="kv"><b>learner — ${esc(n.learner)}</b><div class="q">${esc(n.learner_goal)}</div></div>
    <div class="kv"><b>partner — ${esc(n.partner)} (never shown the key's contents as text)</b>
      <div class="q">${esc(n.partner_goal)}</div></div>
  </div>

  ${D.others.length?`<h2><span class="n">3</span>The same scenario, two learners</h2>
  <p class="sub">Identical scenario, identical partner and judge — only the learner changes. This is the
     discrimination result made concrete.</p>${discrim()}`:''}

  <h2><span class="n">${D.others.length?4:3}</span>Transcripts</h2>
  <p class="sub">The rail on the right shows what the judge actually recorded: which conditions were
     met, which triggers were tripped, and whether the internal state was reached.</p>
  ${viewer()}

  <details class="raw"><summary>raw partner_key JSON (both scenarios)</summary>
    <pre>${esc(JSON.stringify({old:o.raw_key,new:n.raw_key},null,2))}</pre></details>`;
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
