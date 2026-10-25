"""Utilitários compartilhados pelos experimentos."""

from __future__ import annotations

import csv
import os
import shutil
import statistics
import time
from pathlib import Path

import ray

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SCRATCH = ROOT / ".scratch"


def init_ray(num_cpus: int = 4) -> None:
    """Sobe um cluster Ray local, silencioso e reprodutível."""
    if ray.is_initialized():
        return
    ray.init(
        num_cpus=num_cpus,
        include_dashboard=False,
        log_to_driver=False,
        configure_logging=True,
        logging_level="error",
        # Garante que os workers consigam importar o pacote `rayckpt`.
        runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}},
    )


def fresh_workdir(name: str) -> Path:
    """Diretório de checkpoint limpo (cada execução parte do zero)."""
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  -> {path.relative_to(ROOT)} ({len(rows)} linhas)")


def measure_rpc_latency(agent, n: int = 200) -> float:
    """Latência mediana de uma chamada de actor sem trabalho algum.

    Serve para separar o custo de RPC do Ray do trabalho da aplicação — sem
    isso, intervalos pequenos pareceriam mais caros do que são.
    """
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        ray.get(agent.probe.remote())
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)
