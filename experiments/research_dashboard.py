"""Generate a self-contained static research status/results dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.research_results import aggregate_research_results


def build_research_dashboard(*, output_root: str = "result") -> Path:
    """Build an offline HTML dashboard from canonical CSV artifacts."""

    aggregate_research_results(output_root=output_root)
    root = Path(output_root)
    results_root = root / "aggregates"
    payload = {
        "status": _read_csv(results_root / "experiment_status.csv"),
        "results": _read_csv(results_root / "results_mean_std.csv"),
        "main": _read_csv(results_root / "main_results.csv"),
        "structured": _read_csv(results_root / "structured_results.csv"),
        "ablation": _read_csv(results_root / "ablation_results.csv"),
        "knowledge": _read_csv(results_root / "knowledge_results.csv"),
        "coverage": _read_csv(results_root / "coverage_results.csv"),
        "significance": _read_csv(results_root / "significance_results.csv"),
        "external": _read_csv(results_root / "literature_reported_results.csv"),
        "preflight": _read_json(root / "research_planning" / "protocol_preflight.json"),
        "leakage": _read_json(root / "research_planning" / "fhm_leakage_audit.json"),
        "paper": _read_json(Path("latex/generated/generation_manifest.json")),
    }
    dashboard = root / "dashboard" / "index.html"
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.write_text(_html(payload), encoding="utf-8")
    return dashboard


def _html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HarMeme to FHM Research Dashboard</title>
<style>
:root{{--bg:#f5f7f8;--ink:#182024;--muted:#5e6b70;--line:#cfd7da;--panel:#fff;--accent:#136f63;--warn:#a34b00}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:#17252a;color:#fff;padding:20px 28px}}h1{{margin:0;font-size:24px;letter-spacing:0}}header p{{margin:5px 0 0;color:#c8d5d8}}
nav{{display:flex;gap:4px;padding:12px 28px 0}}button{{border:1px solid var(--line);background:#fff;padding:8px 12px;cursor:pointer}}button.active{{background:var(--accent);color:#fff}}
main{{padding:16px 28px 40px}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}}
.stat{{background:var(--panel);border:1px solid var(--line);padding:13px;border-radius:4px}}.stat strong{{display:block;font-size:21px}}
.panel{{display:none;background:var(--panel);border:1px solid var(--line);overflow:auto}}.panel.active{{display:block}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{border-bottom:1px solid #e4e8ea;text-align:left;padding:8px 10px}}th{{position:sticky;top:0;background:#edf2f3}}
.empty{{padding:28px;color:var(--muted)}}.fail{{color:#a62222;font-weight:650}}.pass{{color:var(--accent);font-weight:650}}
</style></head><body><header><h1>HarMeme to FHM Research Dashboard</h1><p>Locked source-train / held-out-target protocol. Missing results are shown as status, never as zero.</p></header>
<nav><button data-tab="overview" class="active">Overview</button><button data-tab="protocol">Dataset protocol</button><button data-tab="main">Main performance</button><button data-tab="structured">Structured</button><button data-tab="ablation">Ablation</button><button data-tab="knowledge">Knowledge</button><button data-tab="evidence">Evidence &amp; rationale</button><button data-tab="errors">Errors</button><button data-tab="external">External readiness</button><button data-tab="paper">Paper readiness</button></nav>
<main><div id="summary" class="summary"></div><section id="overview" class="panel active"></section><section id="protocol" class="panel"></section><section id="main" class="panel"></section><section id="structured" class="panel"></section><section id="ablation" class="panel"></section><section id="knowledge" class="panel"></section><section id="evidence" class="panel"></section><section id="errors" class="panel"></section><section id="external" class="panel"></section><section id="paper" class="panel"></section></main>
<script>const DATA={data};
function esc(v){{return String(v??'--').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]))}}
function table(rows){{if(!rows.length)return '<div class="empty">No completed data available.</div>';const cols=[...new Set(rows.flatMap(Object.keys))];return '<table><thead><tr>'+cols.map(c=>'<th>'+esc(c)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+esc(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}}
document.getElementById('overview').innerHTML=table(DATA.status||[]);
document.getElementById('protocol').innerHTML=table([{{protocol:DATA.preflight?.registry?.protocol?.name||'not run',source_train:JSON.stringify(DATA.preflight?.manifests?.source?.statistics?.train||{{}}),source_validation:JSON.stringify(DATA.preflight?.manifests?.source?.statistics?.validation||{{}}),fhm_test:JSON.stringify(DATA.preflight?.manifests?.fhm?.statistics||{{}}),memotion_disabled:DATA.leakage?.checks?.memotion_disabled,split_hash:DATA.preflight?.manifests?.source_manifest_sha256,leakage_status:DATA.leakage?.status}}]);
document.getElementById('main').innerHTML=table(DATA.main||[]);
document.getElementById('structured').innerHTML=table([...(DATA.structured||[]),...(DATA.coverage||[])]);
document.getElementById('ablation').innerHTML=table(DATA.ablation||[]);
document.getElementById('knowledge').innerHTML=table(DATA.knowledge||[]);
document.getElementById('evidence').innerHTML='<div class="empty">Automatic proxy and human-evaluation records appear after audited exports. Silver labels and proxies are identified by provenance.</div>';
document.getElementById('errors').innerHTML='<div class="empty">Error packages appear under result/analysis/research_error_cases after completed FHM runs.</div>';
document.getElementById('external').innerHTML=table(DATA.external||[]);
document.getElementById('paper').innerHTML=table([DATA.paper||{{status:'not exported'}}]);
const completed=(DATA.status||[]).filter(r=>r.status==='complete').length,blocked=(DATA.status||[]).filter(r=>String(r.status).startsWith('blocked')).length;
document.getElementById('summary').innerHTML=`<div class="stat"><span>Registered/run rows</span><strong>${{(DATA.status||[]).length}}</strong></div><div class="stat"><span>Completed</span><strong>${{completed}}</strong></div><div class="stat"><span>Blocked</span><strong>${{blocked}}</strong></div><div class="stat"><span>Protocol preflight</span><strong class="${{DATA.preflight.status==='pass'?'pass':'fail'}}">${{esc(DATA.preflight.status||'not run')}}</strong></div>`;
document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')}});
</script></body></html>"""


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = ["build_research_dashboard"]
