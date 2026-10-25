"""
Orquestra os experimentos na ordem e gera as figuras.

Cada experimento roda em um processo separado para que o cluster Ray seja
derrubado entre eles — um actor que morreu no experimento anterior não pode
influenciar o seguinte. O ruído do raylet é filtrado da saída (mas guardado
em results/run.log, porque é a evidência de que as falhas foram reais).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .common import RESULTS, ROOT

# Linhas de log do sistema que são esperadas (nós matamos actors de propósito)
NOISE = (
    "(raylet)",
    "A worker died or was killed",
    "Task ToolCallingAgent.step failed",
    "The actor is temporarily unavailable",
    "Worker exit type: SYSTEM_ERROR",
)

STEPS = [
    ("A — overhead de checkpoint vs. intervalo", "rayckpt.exp_overhead"),
    ("B — trabalho perdido vs. intervalo", "rayckpt.exp_failure"),
    ("D — validação ponta a ponta sob falhas de Poisson", "rayckpt.exp_endtoend"),
]


def run_step(title: str, module: str, extra: list[str], log) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    log.write(f"\n===== {title} ({module}) =====\n")
    log.flush()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", module, *extra],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.write(line)
        if not any(n in line for n in NOISE):
            print(line, end="", flush=True)
    rc = proc.wait()
    log.flush()
    dt = time.perf_counter() - t0
    print(f"  ({dt / 60:.1f} min)", flush=True)
    if rc != 0:
        raise SystemExit(f"{module} falhou com código {rc} — ver results/run.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="versão curta, só para verificar")
    ap.add_argument("--skip", nargs="*", default=[], help="módulos a pular, ex: exp_overhead")
    args = ap.parse_args()

    extra = ["--quick"] if args.quick else []
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with open(RESULTS / "run.log", "w") as log:
        for title, module in STEPS:
            if module.split(".")[-1] in args.skip:
                print(f"\n(pulando {module})", flush=True)
                continue
            run_step(title, module, extra, log)

        run_step("C — análise, modelo de custo e figuras", "rayckpt.analysis", [], log)
        run_step("Slides da apresentação", "rayckpt.make_slides", [], log)

    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"Dados brutos em  results/")
    print(f"Figuras em       figures/")
    print(f"Resumo em        results/summary.md")
    print(f"Log completo em  results/run.log")


if __name__ == "__main__":
    main()
