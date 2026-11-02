"""python -m portforge_tt.kernel {evaluate,run,report} ...

  evaluate  seed (and import) kernels, run the trusted gates and one bench; no agents
  run       a full campaign: rounds of plan / implement / gate / measure
  certify   G5 for the run's frontier: second session, held-out regimes, SMT
  report    markdown report of a run directory
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

from . import report as report_mod
from .gates import RemoteGates

REPO = Path(__file__).resolve().parents[2]


def _pairs(values):
    out = {}
    for v in values or []:
        label, _, path = v.partition("=")
        out[label] = path
    return out


def _campaign(args):
    from .orchestrator import Campaign
    run_dir = Path(args.run) if args.run else REPO / "runs" / "kernel" / (
        f"{args.family}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    families = json.loads((REPO / args.pack / "families.json").read_text())["families"]
    entry = families[args.family]
    doc = json.loads((REPO / args.pack / "kernels" / args.family / "TECHNIQUES.json").read_text())
    gates = RemoteGates(args.host, args.remote_workdir, run_dir.name)
    gates.sync_code(args.pack)
    camp = Campaign(run_dir, args.pack, args.family, gates, entry, doc, workers=args.workers, beam=args.beam,
                    plans=args.plans, planner=args.planner, rounds=args.rounds, hours=args.hours,
                    board_seconds=args.board_minutes * 60, implementer_minutes=args.minutes,
                    bench_rounds=args.bench_rounds, log=lambda *a: print(*a, flush=True))
    ctx = run_dir / "context"
    if not ctx.exists():
        ctx.mkdir(parents=True)
        r = gates.call("context", {"pack": args.pack, "family": args.family})
        for name, text in (r.get("sources") or {}).items():
            (ctx / (name if "." in name else f"{name}.c")).write_text(text)
    return camp, run_dir


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m portforge_tt.kernel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "run", "certify"):
        s = sub.add_parser(name)
        s.add_argument("--pack", default="ports/xnnpack-rvv")
        s.add_argument("--family", required=True)
        s.add_argument("--seed-kernel", action="append", help="label=path (repeatable)")
        s.add_argument("--import", dest="imports", action="append", help="label=path of archived variants")
        s.add_argument("--run", help="run directory (default runs/kernel/<family>-<utc>)")
        s.add_argument("--host", default="tt_box")
        s.add_argument("--remote-workdir", default="portforge-cpu", help="relative to the remote home")
        s.add_argument("--workers", type=int, default=2)
        s.add_argument("--beam", type=int, default=3)
        s.add_argument("--plans", type=int, default=4)
        s.add_argument("--planner", choices=["thompson", "llm"], default="thompson")
        s.add_argument("--rounds", type=int, default=5)
        s.add_argument("--hours", type=float, default=6.0)
        s.add_argument("--board-minutes", type=float, default=120)
        s.add_argument("--minutes", type=int, default=20, help="implementer time cap")
        s.add_argument("--bench-rounds", type=int, default=15)
        s.add_argument("--patience", type=int, default=3, help="rounds without a 1%% frontier gain before stopping")
        s.add_argument("--agent", choices=["claude", "opencode", "codex"], default="claude")
        s.add_argument("--placeholder", help="path of a stub source: port from scratch (no RVV exists) "
                                              "before optimizing")
        s.add_argument("--ports", type=int, default=4, help="independent ports per porting round")
    s = sub.add_parser("report")
    s.add_argument("run")
    args = p.parse_args(argv)
    if args.cmd == "report":
        text = report_mod.render(args.run)
        (Path(args.run) / "REPORT.md").write_text(text)
        print(text)
        return 0
    os.environ.setdefault("PORTFORGE_WORKER_BACKEND", args.agent)
    os.environ.setdefault("PORTFORGE_SUPERVISOR_BACKEND", args.agent)
    camp, run_dir = _campaign(args)
    if args.placeholder:
        camp.seed({"placeholder": args.placeholder}, "placeholder")
    camp.seed(_pairs(args.seed_kernel), "seed")
    camp.seed(_pairs(args.imports), "import")
    if args.cmd == "run":
        if args.placeholder:
            camp.port(ports=args.ports)
        camp.run(patience=args.patience)
    if args.cmd == "certify":
        camp.certify()
    status = camp.status("evaluated" if args.cmd == "evaluate" else "finished")
    (run_dir / "REPORT.md").write_text(report_mod.render(run_dir))
    print(json.dumps(status, indent=1, default=str))
    print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
