# 功能：解析 PaddleDetection 训练日志，生成可自动刷新的本地训练监控页。
from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(
    r"Epoch:\s*\[(?P<epoch>\d+)\]\s*\[\s*(?P<step>\d+)\s*/\s*(?P<steps>\d+)\]\s*"
    r"learning_rate:\s*(?P<lr>[0-9.eE+-]+)\s*"
    r"loss:\s*(?P<loss>[0-9.eE+-]+)\s*"
    r"loss_cls:\s*(?P<loss_cls>[0-9.eE+-]+)\s*"
    r"loss_iou:\s*(?P<loss_iou>[0-9.eE+-]+)\s*"
    r"loss_dfl:\s*(?P<loss_dfl>[0-9.eE+-]+)\s*"
    r"loss_l1:\s*(?P<loss_l1>[0-9.eE+-]+).*?"
    r"eta:\s*(?P<eta>.*?)\s*"
    r"batch_cost:\s*(?P<batch_cost>[0-9.eE+-]+)\s*"
    r"data_cost:\s*(?P<data_cost>[0-9.eE+-]+)\s*"
    r"ips:\s*(?P<ips>[0-9.eE+-]+)\s*images/s"
)
AP_RE = re.compile(
    r"Average Precision\s*\(AP\)\s*@\[ IoU=(?P<iou>.*?)\| area=\s*(?P<area>.*?)\| maxDets=(?P<maxdets>.*?)\]\s*=\s*(?P<value>[0-9.]+)"
)
AR_RE = re.compile(
    r"Average Recall\s*\(AR\)\s*@\[ IoU=(?P<iou>.*?)\| area=\s*(?P<area>.*?)\| maxDets=(?P<maxdets>.*?)\]\s*=\s*(?P<value>[0-9.]+)"
)
BEST_RE = re.compile(r"Best test bbox ap is (?P<value>[0-9]+(?:\.[0-9]+)?)")
SAVE_RE = re.compile(r"Save checkpoint:\s*(?P<path>.+)$")
EVAL_START_RE = re.compile(r"Eval iter:\s*0\b")
WARNING_RE = re.compile(r"(Found inf or nan|Traceback|RuntimeError|Exception|Error:)", re.IGNORECASE)
FATAL_RE = re.compile(
    r"(Traceback \(most recent call last\):|RuntimeError:|FatalError:|OutOfMemoryError|"
    r"CUDA out of memory|CUDA error:|CUDNN_STATUS_[A-Z_]+|MemoryError:)",
    re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="PaddleDetection stdout log path.")
    parser.add_argument("--error-log", type=Path, default=None, help="Optional PaddleDetection stderr log path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for training_monitor.html and JSON.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Expected total epochs.")
    parser.add_argument("--interval", type=float, default=10.0, help="Watch update interval in seconds.")
    parser.add_argument("--watch", action="store_true", help="Keep refreshing until interrupted.")
    parser.add_argument("--serve", action="store_true", help="Serve the monitor directory over HTTP while refreshing.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host for --serve.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port for --serve.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.serve:
        return serve_monitor(
            args.log,
            args.output_dir,
            args.max_epochs,
            args.interval,
            args.host,
            args.port,
            error_log_path=args.error_log,
        )
    if args.watch:
        while True:
            write_monitor(args.log, args.output_dir, args.max_epochs, error_log_path=args.error_log)
            time.sleep(args.interval)
    write_monitor(args.log, args.output_dir, args.max_epochs, error_log_path=args.error_log)
    return 0


def write_monitor(
    log_path: Path,
    output_dir: Path,
    max_epochs: int,
    *,
    error_log_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = parse_log(log_path, max_epochs=max_epochs, error_log_path=error_log_path)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_monitor.html").write_text(render_html(metrics), encoding="utf-8")
    return metrics


def serve_monitor(
    log_path: Path,
    output_dir: Path,
    max_epochs: int,
    interval: float,
    host: str,
    port: int,
    *,
    error_log_path: Path | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    def refresher() -> None:
        while not stop_event.is_set():
            write_monitor(log_path, output_dir, max_epochs, error_log_path=error_log_path)
            stop_event.wait(interval)

    thread = threading.Thread(target=refresher, name="monitor-refresher", daemon=True)
    thread.start()
    write_monitor(log_path, output_dir, max_epochs, error_log_path=error_log_path)

    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    with ThreadingHTTPServer((host, port), handler) as server:
        actual_host, actual_port = server.server_address
        print(f"Serving {output_dir} at http://{actual_host}:{actual_port}/training_monitor.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
    return 0


def parse_log(
    log_path: Path,
    *,
    max_epochs: int,
    error_log_path: Path | None = None,
) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
    error_lines = (
        error_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if error_log_path is not None and error_log_path.exists()
        else []
    )
    train_points: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    current_eval: dict[str, Any] | None = None
    checkpoint_saves: list[str] = []
    warnings: list[dict[str, str | None]] = []
    best_ap = None
    failure: dict[str, Any] | None = None

    for source, source_lines in (("stdout", lines), ("stderr", error_lines)):
        for line_number, line in enumerate(source_lines, start=1):
            if WARNING_RE.search(line):
                warnings.append({"time": _line_time(line), "message": line.strip(), "source": source})
            fatal_match = FATAL_RE.search(line)
            if fatal_match:
                message = line.strip()
                failure = {
                    "id": f"{source}:{line_number}:{message}",
                    "source": source,
                    "line_number": line_number,
                    "time": _line_time(line),
                    "kind": fatal_match.group(0),
                    "message": message,
                }

    for line in lines:
        train_match = TRAIN_RE.search(line)
        if train_match:
            point = {
                "raw": line,
                "time": _line_time(line),
                "epoch": int(train_match.group("epoch")),
                "step": int(train_match.group("step")),
                "steps_per_epoch": int(train_match.group("steps")),
                "learning_rate": float(train_match.group("lr")),
                "loss": float(train_match.group("loss")),
                "loss_cls": float(train_match.group("loss_cls")),
                "loss_iou": float(train_match.group("loss_iou")),
                "loss_dfl": float(train_match.group("loss_dfl")),
                "loss_l1": float(train_match.group("loss_l1")),
                "eta": train_match.group("eta").strip(),
                "batch_cost": float(train_match.group("batch_cost")),
                "data_cost": float(train_match.group("data_cost")),
                "ips": float(train_match.group("ips")),
            }
            point["global_step"] = point["epoch"] * point["steps_per_epoch"] + point["step"]
            point["progress_percent"] = _progress_percent(point, max_epochs)
            train_points.append(point)
            continue

        if EVAL_START_RE.search(line):
            current_eval = {"time": _line_time(line), "ap": {}, "ar": {}}
            continue

        ap_match = AP_RE.search(line)
        if ap_match:
            if current_eval is None:
                current_eval = {"time": _line_time(line), "ap": {}, "ar": {}}
            key = _metric_key(ap_match)
            current_eval["ap"][key] = float(ap_match.group("value"))
            continue

        ar_match = AR_RE.search(line)
        if ar_match:
            if current_eval is None:
                current_eval = {"time": _line_time(line), "ap": {}, "ar": {}}
            key = _metric_key(ar_match)
            current_eval["ar"][key] = float(ar_match.group("value"))
            continue

        best_match = BEST_RE.search(line)
        if best_match:
            best_ap = float(best_match.group("value"))
            if current_eval is not None:
                current_eval["best_ap"] = best_ap
                current_eval["time"] = current_eval.get("time") or _line_time(line)
                evals.append(current_eval)
                current_eval = None
            continue

        save_match = SAVE_RE.search(line)
        if save_match:
            checkpoint_saves.append(save_match.group("path").strip())

    latest = train_points[-1] if train_points else None
    steps_per_epoch = latest["steps_per_epoch"] if latest else None
    total_steps = (steps_per_epoch * max_epochs) if steps_per_epoch else None
    remaining_steps = (total_steps - latest["global_step"]) if latest and total_steps else None
    completed = bool(latest and latest["epoch"] >= max_epochs - 1 and evals)
    if completed:
        latest["progress_percent"] = 100.0
        remaining_steps = 0
    checkpoint_files = []
    if log_path.exists():
        checkpoint_dir = log_path.parent
        for pattern in ("*.pdparams", "*.pdema", "*.pdopt"):
            checkpoint_files.extend(p for p in checkpoint_dir.glob(pattern) if p.is_file())
        checkpoint_files = sorted({p.resolve() for p in checkpoint_files}, key=lambda p: p.stat().st_mtime, reverse=True)[:12]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": str(log_path),
        "error_log_path": str(error_log_path) if error_log_path is not None else None,
        "max_epochs": max_epochs,
        "latest": latest,
        "total_train_points": len(train_points),
        "train_points": train_points,
        "evals": evals,
        "latest_eval": evals[-1] if evals else None,
        "best_ap": best_ap,
        "checkpoint_saves": checkpoint_saves[-5:],
        "checkpoint_files": [str(p) for p in checkpoint_files],
        "warnings": warnings[-20:],
        "failure": failure,
        "total_steps": total_steps,
        "remaining_steps": remaining_steps,
        "status": (
            "failed"
            if failure is not None
            else ("completed" if completed else ("running_or_recent_log" if latest else "waiting_for_log"))
        ),
    }


def render_html(metrics: dict[str, Any]) -> str:
    initial_json = json.dumps(metrics, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Target Det Training Monitor</title>
  <link rel="icon" href="data:,">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --ink: #172033;
      --muted: #657084;
      --panel: #ffffff;
      --border: #dde3ee;
      --green: #18a058;
      --blue: #2563eb;
      --red: #dc2626;
      --amber: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 20px 28px;
      color: #fff;
      background: #111827;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    main {{ padding: 22px 28px 40px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    header .muted {{ color: #d1d5db; overflow-wrap: anywhere; }}
    .topline {{ display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 5px 9px;
      border: 1px solid rgba(255,255,255,.35);
      border-radius: 8px;
      font-size: 13px;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,.16);
    }}
    .status-pill.failed {{ border-color: #fca5a5; background: rgba(220,38,38,.18); }}
    .status-pill.failed .dot {{ background: #ef4444; box-shadow: 0 0 0 4px rgba(239,68,68,.2); }}
    .failure-banner {{
      margin-top: 0;
      border-color: #fca5a5;
      background: #fff1f2;
      color: #991b1b;
    }}
    .failure-banner[hidden] {{ display: none; }}
    .failure-banner h2 {{ color: #b91c1c; }}
    .failure-detail {{ margin: 7px 0 0; font-family: Consolas, monospace; overflow-wrap: anywhere; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15,23,42,.05);
    }}
    .card {{ padding: 14px; min-height: 76px; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 7px; }}
    .value {{ font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }}
    section {{ padding: 16px; margin: 16px 0; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .bar {{ height: 18px; background: #e5e7eb; border-radius: 8px; overflow: hidden; }}
    .bar > div {{ height: 100%; width: 0%; background: var(--green); transition: width .25s ease; }}
    svg {{ width: 100%; height: 290px; display: block; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }}
    .logline {{
      margin-top: 10px;
      padding: 10px;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      font-family: Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .warn {{ color: var(--amber); font-weight: 700; }}
    .error {{ color: var(--red); font-weight: 700; }}
    @media (max-width: 640px) {{
      header {{ padding: 16px; }}
      main {{ padding: 14px 12px 28px; min-width: 0; }}
      h1 {{ font-size: 21px; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
      .card {{ padding: 11px; min-width: 0; }}
      .value {{ font-size: 19px; }}
      .charts {{ grid-template-columns: minmax(0, 1fr); gap: 10px; }}
      section {{ padding: 12px; min-width: 0; }}
      svg {{ height: auto; min-height: 220px; }}
      th, td {{ padding: 7px 6px; overflow-wrap: anywhere; }}
      ul {{ padding-left: 20px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <h1>01_target_det 训练实时监控</h1>
      <span class="status-pill" id="statusPill"><span class="dot"></span><span id="statusText">读取中</span></span>
    </div>
    <div class="muted">页面每 5 秒自动读取 <code>training_metrics.json</code>，无需手动刷新。</div>
    <div class="muted" id="logPath"></div>
  </header>
  <main>
    <section id="failureBanner" class="failure-banner" role="alert" aria-live="assertive" hidden>
      <h2>训练失败</h2>
      <div>监控检测到确定性致命错误，训练结果可能已经停止更新。</div>
      <div class="failure-detail" id="failureMessage"></div>
    </section>
    <div class="grid" id="cards"></div>
    <section>
      <h2>整体进度</h2>
      <div class="bar"><div id="progressBar"></div></div>
      <p class="muted" id="progressText">等待日志。</p>
      <div class="logline" id="latestRaw">暂无最新训练日志。</div>
    </section>
    <div class="charts">
      <section><h2>Loss 曲线</h2><div id="lossChart"></div></section>
      <section><h2>IPS 曲线</h2><div id="ipsChart"></div></section>
      <section><h2>验证 AP 曲线</h2><div id="apChart"></div></section>
      <section><h2>验证 Small / Medium / Large AP</h2><div id="areaApChart"></div></section>
    </div>
    <section><h2>最近一次验证指标</h2><div id="evalTable"></div></section>
    <section><h2>最近 checkpoint</h2><div id="checkpointList"></div></section>
    <section>
      <h2>最近告警</h2>
      <div id="warningBox" class="muted">暂无。</div>
    </section>
  </main>
  <script id="initialMetrics" type="application/json">{initial_json}</script>
  <script>
    const REFRESH_MS = 5000;
    const NORMAL_TITLE = 'Target Det Training Monitor';
    const FAILURE_ALERT_KEY = 'target-det-training-failure-alert';
    const initialMetrics = JSON.parse(document.getElementById('initialMetrics').textContent);
    let lastGeneratedAt = null;

    function fmt(value, digits = 4) {{
      const number = Number(value);
      if (!Number.isFinite(number)) return '-';
      return number.toFixed(digits);
    }}

    function metricKey(metrics, family, key) {{
      const latestEval = metrics.latest_eval || {{}};
      return Number(latestEval[family]?.[key]);
    }}

    function escapeHtml(text) {{
      return String(text ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function drawLineChart(values, options) {{
      if (!values || values.length < 2) {{
        return '<p class="muted">等待更多数据点。</p>';
      }}
      const width = 900;
      const height = 290;
      const padL = 58;
      const padR = 24;
      const padT = 24;
      const padB = 36;
      const xs = values.map(v => v[0]);
      const ys = values.map(v => v[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = options.yMin ?? Math.min(...ys);
      const maxY = options.yMax ?? Math.max(...ys);
      const ySpan = Math.max(1e-9, maxY - minY);
      const xSpan = Math.max(1e-9, maxX - minX);
      const sx = x => padL + (x - minX) / xSpan * (width - padL - padR);
      const sy = y => padT + (maxY - y) / ySpan * (height - padT - padB);
      const points = values.map(v => `${{sx(v[0]).toFixed(2)}},${{sy(v[1]).toFixed(2)}}`).join(' ');
      const latest = values[values.length - 1][1];
      const ticks = [0, .25, .5, .75, 1].map(r => minY + (maxY - minY) * r);
      const grid = ticks.map(t => {{
        const y = sy(t);
        return `<line x1="${{padL}}" y1="${{y}}" x2="${{width-padR}}" y2="${{y}}" stroke="#e5e7eb"/>` +
               `<text x="6" y="${{y+4}}" fill="#6b7280" font-size="12">${{t.toFixed(options.digits ?? 3)}}</text>`;
      }}).join('');
      return `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{escapeHtml(options.label)}} chart">
        ${{grid}}
        <line x1="${{padL}}" y1="${{height-padB}}" x2="${{width-padR}}" y2="${{height-padB}}" stroke="#9ca3af"/>
        <line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{height-padB}}" stroke="#9ca3af"/>
        <polyline points="${{points}}" fill="none" stroke="${{options.color}}" stroke-width="3"/>
        <circle cx="${{sx(values[values.length-1][0]).toFixed(2)}}" cy="${{sy(latest).toFixed(2)}}" r="4" fill="#111827"/>
        <text x="${{padL}}" y="17" fill="#374151" font-size="13">${{escapeHtml(options.label)}} latest=${{fmt(latest, options.digits ?? 4)}}</text>
      </svg>`;
    }}

    function renderEvalTable(metrics) {{
      const latestEval = metrics.latest_eval;
      if (!latestEval) return '<p class="muted">尚未完成验证。</p>';
      const rows = [];
      for (const family of ['ap', 'ar']) {{
        const group = latestEval[family] || {{}};
        for (const [key, value] of Object.entries(group)) {{
          rows.push(`<tr><td>${{family.toUpperCase()}}</td><td>${{escapeHtml(key)}}</td><td>${{fmt(value, 3)}}</td></tr>`);
        }}
      }}
      if (latestEval.best_ap !== undefined) {{
        rows.push(`<tr><td>BEST</td><td>bbox AP</td><td>${{fmt(latestEval.best_ap, 3)}}</td></tr>`);
      }}
      return '<table><thead><tr><th>类型</th><th>指标</th><th>值</th></tr></thead><tbody>' + rows.join('') + '</tbody></table>';
    }}

    function renderCheckpoints(metrics) {{
      const saves = metrics.checkpoint_files || metrics.checkpoint_saves || [];
      if (!saves.length) return '<ul><li>暂无</li></ul>';
      return '<ul>' + saves.map(path => `<li>${{escapeHtml(path)}}</li>`).join('') + '</ul>';
    }}

    function renderWarnings(metrics) {{
      const warnings = metrics.warnings || [];
      if (!warnings.length) return '<span class="muted">暂无。</span>';
      return '<ul>' + warnings.slice(-8).map(item => `<li><span class="warn">${{escapeHtml(item.time || '')}}</span> ${{escapeHtml(item.message || item.raw || '')}}</li>`).join('') + '</ul>';
    }}

    function renderFailure(metrics) {{
      const failure = metrics.failure;
      const failed = metrics.status === 'failed' && Boolean(failure);
      const banner = document.getElementById('failureBanner');
      const statusPill = document.getElementById('statusPill');
      banner.hidden = !failed;
      statusPill.classList.toggle('failed', failed);
      if (!failed) {{
        document.title = NORMAL_TITLE;
        return;
      }}

      const source = failure.source || 'unknown';
      const lineNumber = failure.line_number ?? '-';
      const message = failure.message || '检测到未知致命错误。';
      document.getElementById('failureMessage').textContent =
        `${{source}} 第 ${{lineNumber}} 行：${{message}}`;
      document.title = `训练失败 | ${{NORMAL_TITLE}}`;

      const failureId = String(failure.id || `${{source}}:${{lineNumber}}:${{message}}`);
      try {{
        if (sessionStorage.getItem(FAILURE_ALERT_KEY) !== failureId) {{
          sessionStorage.setItem(FAILURE_ALERT_KEY, failureId);
          setTimeout(() => window.alert(`训练失败\n${{message}}`), 0);
        }}
      }} catch (error) {{
        console.warn('无法记录失败提醒状态', error);
      }}
    }}

    function render(metrics) {{
      const latest = metrics.latest || {{}};
      const progress = Number(latest.progress_percent || 0);
      const maxEpoch = Number(metrics.max_epochs || 0);
      const cards = [
        ['当前 Epoch', `${{latest.epoch ?? '-'}}/${{maxEpoch ? maxEpoch - 1 : '-'}}`],
        ['当前 Step', `${{latest.step ?? '-'}}/${{latest.steps_per_epoch ?? '-'}}`],
        ['整体进度', `${{fmt(progress, 2)}}%`],
        ['当前 Loss', fmt(latest.loss, 4)],
        ['当前 IPS', fmt(latest.ips, 2)],
        ['ETA', latest.eta || '-'],
        ['当前 LR', fmt(latest.learning_rate, 6)],
        ['Best AP', fmt(metrics.best_ap, 3)],
        ['最新验证 AP', fmt(metricKey(metrics, 'ap', 'IoU=0.50:0.95 area=all maxDets=100'), 3)],
        ['Small AP', fmt(metricKey(metrics, 'ap', 'IoU=0.50:0.95 area=small maxDets=100'), 3)],
        ['Medium AP', fmt(metricKey(metrics, 'ap', 'IoU=0.50:0.95 area=medium maxDets=100'), 3)],
        ['Large AP', fmt(metricKey(metrics, 'ap', 'IoU=0.50:0.95 area=large maxDets=100'), 3)],
      ];
      document.getElementById('cards').innerHTML = cards.map(([name, value]) =>
        `<div class="card"><div class="label">${{escapeHtml(name)}}</div><div class="value">${{escapeHtml(value)}}</div></div>`
      ).join('');
      document.getElementById('progressBar').style.width = `${{Math.max(0, Math.min(100, progress))}}%`;
      document.getElementById('progressText').textContent =
        `${{fmt(progress, 2)}}% | 最新记录：Epoch ${{latest.epoch ?? '-'}} Step ${{latest.step ?? '-'}} | 指标生成时间：${{metrics.generated_at || '-'}}`;
      document.getElementById('latestRaw').textContent = latest.raw || '暂无最新训练日志。';
      document.getElementById('logPath').textContent =
        `训练日志：${{metrics.log_path || '-'}} | 错误日志：${{metrics.error_log_path || '-'}}`;
      document.getElementById('statusText').textContent = metrics.status === 'failed'
        ? '训练失败'
        : (metrics.status === 'completed' ? '训练已完成' : `实时刷新中 | ${{metrics.status || '-'}}`);
      renderFailure(metrics);

      const trainPoints = metrics.train_points || [];
      document.getElementById('lossChart').innerHTML = drawLineChart(
        trainPoints.map(p => [Number(p.progress_percent), Number(p.loss)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])),
        {{ label: 'Loss', color: 'var(--blue)', digits: 4 }}
      );
      document.getElementById('ipsChart').innerHTML = drawLineChart(
        trainPoints.map(p => [Number(p.progress_percent), Number(p.ips)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])),
        {{ label: 'IPS', color: 'var(--green)', digits: 2 }}
      );
      const evals = metrics.evals || [];
      document.getElementById('apChart').innerHTML = drawLineChart(
        evals.map((e, i) => [i + 1, Number(e.ap?.['IoU=0.50:0.95 area=all maxDets=100'])]).filter(p => Number.isFinite(p[1])),
        {{ label: 'AP50:95', color: 'var(--red)', yMin: 0.85, yMax: 1.0, digits: 3 }}
      );
      const areaValues = [];
      for (const [name, color, key] of [
        ['small', '#b7791f', 'IoU=0.50:0.95 area=small maxDets=100'],
        ['medium', '#2563eb', 'IoU=0.50:0.95 area=medium maxDets=100'],
        ['large', '#18a058', 'IoU=0.50:0.95 area=large maxDets=100'],
      ]) {{
        const values = evals.map((e, i) => [i + 1, Number(e.ap?.[key])]).filter(p => Number.isFinite(p[1]));
        if (values.length >= 2) areaValues.push([name, color, values]);
      }}
      document.getElementById('areaApChart').innerHTML = areaValues.length
        ? drawMultiLineChart(areaValues, {{ yMin: 0.5, yMax: 1.0, digits: 3 }})
        : '<p class="muted">尚未完成足够验证。</p>';
      document.getElementById('evalTable').innerHTML = renderEvalTable(metrics);
      document.getElementById('checkpointList').innerHTML = renderCheckpoints(metrics);
      document.getElementById('warningBox').innerHTML = renderWarnings(metrics);
      lastGeneratedAt = metrics.generated_at || lastGeneratedAt;
    }}

    function drawMultiLineChart(series, options) {{
      const width = 900;
      const height = 290;
      const padL = 58, padR = 24, padT = 24, padB = 36;
      const allValues = series.flatMap(s => s[2]);
      const minX = Math.min(...allValues.map(v => v[0]));
      const maxX = Math.max(...allValues.map(v => v[0]));
      const minY = options.yMin ?? Math.min(...allValues.map(v => v[1]));
      const maxY = options.yMax ?? Math.max(...allValues.map(v => v[1]));
      const sx = x => padL + (x - minX) / Math.max(1e-9, maxX - minX) * (width - padL - padR);
      const sy = y => padT + (maxY - y) / Math.max(1e-9, maxY - minY) * (height - padT - padB);
      const polylines = series.map(([name, color, values]) => {{
        const points = values.map(v => `${{sx(v[0]).toFixed(2)}},${{sy(v[1]).toFixed(2)}}`).join(' ');
        const latest = values[values.length - 1];
        return `<polyline points="${{points}}" fill="none" stroke="${{color}}" stroke-width="3"/>
          <circle cx="${{sx(latest[0]).toFixed(2)}}" cy="${{sy(latest[1]).toFixed(2)}}" r="4" fill="${{color}}"/>`;
      }}).join('');
      const legend = series.map(([name, color], i) =>
        `<text x="${{padL + i * 135}}" y="18" fill="${{color}}" font-size="13">${{escapeHtml(name)}}</text>`
      ).join('');
      const grid = [0.5, 0.625, 0.75, 0.875, 1.0].map(t => {{
        const y = sy(t);
        return `<line x1="${{padL}}" y1="${{y}}" x2="${{width-padR}}" y2="${{y}}" stroke="#e5e7eb"/>
          <text x="6" y="${{y+4}}" fill="#6b7280" font-size="12">${{t.toFixed(options.digits ?? 3)}}</text>`;
      }}).join('');
      return `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="area AP chart">
        ${{grid}}
        <line x1="${{padL}}" y1="${{height-padB}}" x2="${{width-padR}}" y2="${{height-padB}}" stroke="#9ca3af"/>
        <line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{height-padB}}" stroke="#9ca3af"/>
        ${{polylines}}
        ${{legend}}
      </svg>`;
    }}

    async function refresh() {{
      try {{
        const response = await fetch(`training_metrics.json?ts=${{Date.now()}}`, {{ cache: 'no-store' }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const metrics = await response.json();
        render(metrics);
      }} catch (error) {{
        document.getElementById('statusText').textContent = `读取失败，显示上次数据`;
        console.warn(error);
      }}
    }}

    render(initialMetrics);
    refresh();
    setInterval(refresh, REFRESH_MS);
  </script>
</body>
</html>
"""


def _line_time(line: str) -> str | None:
    match = re.match(r"\[(?P<stamp>\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\]", line)
    return match.group("stamp") if match else None


def _metric_key(match: re.Match[str]) -> str:
    iou = " ".join(match.group("iou").split())
    area = " ".join(match.group("area").split())
    maxdets = " ".join(match.group("maxdets").split())
    return f"IoU={iou} area={area} maxDets={maxdets}"


def _progress_percent(point: dict[str, Any], max_epochs: int) -> float:
    steps_per_epoch = max(1, int(point["steps_per_epoch"]))
    total = max_epochs * steps_per_epoch
    current = int(point["epoch"]) * steps_per_epoch + int(point["step"])
    return max(0.0, min(100.0, current / total * 100.0))


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}"


def _svg_line_chart(points: list[dict[str, Any]], key: str, label: str, color: str) -> str:
    values = [(float(p["global_step"]), float(p[key])) for p in points if key in p and _is_finite(p[key])]
    if len(values) < 2:
        return "<p class=\"muted\">等待更多训练点。</p>"
    return _render_svg(values, label, color)


def _svg_eval_chart(evals: list[dict[str, Any]]) -> str:
    values = []
    for index, item in enumerate(evals):
        ap = (item.get("ap") or {}).get("IoU=0.50:0.95 area=all maxDets=100")
        if ap is not None:
            values.append((float(index + 1), float(ap)))
    if not values:
        return "<p class=\"muted\">尚未完成验证。</p>"
    return _render_svg(values, "AP50:95", "#dc2626", y_min=0.0, y_max=1.0)


def _render_svg(values: list[tuple[float, float]], label: str, color: str, y_min: float | None = None, y_max: float | None = None) -> str:
    width, height = 900, 260
    pad_l, pad_r, pad_t, pad_b = 55, 20, 20, 35
    xs = [x for x, _ in values]
    ys = [y for _, y in values]
    min_x, max_x = min(xs), max(xs)
    min_y = min(ys) if y_min is None else y_min
    max_y = max(ys) if y_max is None else y_max
    if math.isclose(min_y, max_y):
        min_y -= 1
        max_y += 1
    def sx(x: float) -> float:
        return pad_l + (x - min_x) / (max_x - min_x or 1.0) * (width - pad_l - pad_r)
    def sy(y: float) -> float:
        return pad_t + (max_y - y) / (max_y - min_y or 1.0) * (height - pad_t - pad_b)
    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
    latest = values[-1][1]
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)} chart">
  <line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#9ca3af"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#9ca3af"/>
  <polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>
  <text x="{pad_l}" y="15" fill="#374151" font-size="13">{html.escape(label)} latest={latest:.4f}</text>
  <text x="4" y="{pad_t+5}" fill="#6b7280" font-size="12">{max_y:.4f}</text>
  <text x="4" y="{height-pad_b}" fill="#6b7280" font-size="12">{min_y:.4f}</text>
</svg>"""


def _eval_table(latest_eval: dict[str, Any]) -> str:
    if not latest_eval:
        return "<p class=\"muted\">尚未完成验证。</p>"
    rows = []
    for family in ("ap", "ar"):
        for key, value in (latest_eval.get(family) or {}).items():
            rows.append(f"<tr><td>{family.upper()}</td><td>{html.escape(key)}</td><td>{float(value):.3f}</td></tr>")
    if "best_ap" in latest_eval:
        rows.append(f"<tr><td>BEST</td><td>bbox AP</td><td>{float(latest_eval['best_ap']):.3f}</td></tr>")
    return "<table><thead><tr><th>类型</th><th>指标</th><th>值</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
