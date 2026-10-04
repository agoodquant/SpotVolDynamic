"""python -m spotvol <command>

  backfill [SYM ...]   pull DoltHub history (new dates only)
  snapshot [SYM ...]   store today's Yahoo chain
  build    [SYM ...]   raw quotes -> smiles (new dates only; --rebuild to redo)
  report   SYM [--bench SPY]   write reports/<SYM>.html
  screen               rank the universe by realized spot/vol beta, with today's regime and next-day betas
  backtest SYM         rolling out-of-sample test of the regime model, writes reports/<SYM>_backtest.html
  daily                snapshot + backfill + build + screen for the whole universe

Every command that writes a report also rebuilds reports/index.html.
"""
import argparse
import sys

from . import dolthub, smile, yahoo
from .config import SOURCES, universe


def _symbols(args, u):
    return [s.upper() for s in args.symbols] or u["all"]


def cmd_backfill(syms, u):
    for s in syms:
        try:
            got, failed = dolthub.backfill(s, u["backfill_start"])
            note = f", {failed} dates failed and will be retried on the next run" if failed else ""
            print(f"backfill {s}: {got} new dates{note}", flush=True)
        except Exception as e:
            print(f"backfill {s}: FAILED {e}", flush=True)


def cmd_snapshot(syms):
    for s in syms:
        try:
            session, msg = yahoo.snapshot(s)
            print(f"snapshot {s} {session:%Y-%m-%d}: {msg}" if session is not None else f"snapshot {s}: {msg}", flush=True)
        except Exception as e:
            print(f"snapshot {s}: FAILED {e}", flush=True)


def cmd_build(syms, rebuild=False):
    for s in syms:
        for src in SOURCES:
            n = smile.build(src, s, rebuild)
            if n:
                print(f"build {s} [{src}]: {n} dates", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spotvol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backfill", "snapshot", "build", "report", "screen", "backtest", "daily"])
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    u = universe()
    syms = _symbols(args, u)

    if args.command == "backfill":
        cmd_backfill(syms, u)
    elif args.command == "snapshot":
        cmd_snapshot(syms)
    elif args.command == "build":
        cmd_build(syms, args.rebuild)
    elif args.command == "report":
        from . import report
        if not args.symbols:
            sys.exit("report needs a symbol")
        for s in syms:
            have = [x for x in ([s, args.bench.upper()] if args.bench else [s])]
            cmd_backfill([x for x in have if not _has_data(x)], u)
            cmd_build(have)
            print("wrote", report.write_report(s, args.bench.upper() if args.bench else None))
        _write_index()
    elif args.command == "backtest":
        from . import backtest
        if not args.symbols:
            sys.exit("backtest needs a symbol")
        for s in syms:
            print("wrote", backtest.write_backtest(s))
        _write_index()
    elif args.command == "screen":
        from . import report
        print("wrote", report.write_screen(syms))
        _write_index()
    elif args.command == "daily":
        from . import report
        cmd_snapshot(syms)
        cmd_backfill(syms, u)
        cmd_build(syms)
        print("wrote", report.write_screen(syms))
        _write_index()


def _write_index():
    from . import report
    print("wrote", report.write_index())


def _has_data(symbol):
    from . import store
    return any(store.raw_dates(src, symbol) for src in SOURCES)
