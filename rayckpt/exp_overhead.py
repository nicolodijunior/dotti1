"""
Experimento A — Overhead de checkpoint vs. intervalo (sem falhas).

Pergunta: quanto cada frequência de checkpoint deixa a execução mais lenta?

Método: para cada intervalo N, o driver executa `calls` chamadas sequenciais
no actor (um agente é sequencial por natureza: a próxima tool call depende do
resultado da anterior) e mede o tempo de parede. O intervalo N=0 nunca faz
checkpoint e é a linha de base. O overhead por chamada é a diferença em
relação a essa linha de base.

Repetimos para três tamanhos de estado, porque δ (custo de um checkpoint) é o
parâmetro que desloca o ótimo nas fórmulas de Young e Daly — querermos
mostrar que δ é de fato ~constante por checkpoint e proporcional ao estado.
"""

from __future__ import annotations

import argparse
import time

import ray

from .agent import ToolCallingAgent
from .common import RESULTS, fresh_workdir, init_ray, measure_rpc_latency, write_csv

INTERVALS = [1, 2, 5, 10, 20, 50, 100, 200, 0]  # 0 = nunca faz checkpoint
STATE_SIZES_KB = [64, 512, 4096]


def run_one(
    interval: int,
    state_kb: int,
    calls: int,
    work_s: float,
    warmup: int,
    tag: str,
    ckpt_latency_s: float = 0.0,
    ckpt_mbps: float = 0.0,
) -> dict:
    workdir = fresh_workdir(f"overhead-{tag}")
    agent = ToolCallingAgent.remote(
        workdir=str(workdir),
        checkpoint_every=interval,
        work_s=work_s,
        state_kb=state_kb,
        ckpt_latency_s=ckpt_latency_s,
        ckpt_mbps=ckpt_mbps,
    )
    # Aquece: cria o processo, toca o disco, estabiliza o JIT do interpretador.
    for i in range(warmup):
        ray.get(agent.step.remote(i))
    rpc_s = measure_rpc_latency(agent)

    t0 = time.perf_counter()
    for i in range(warmup, warmup + calls):
        ray.get(agent.step.remote(i))
    wall = time.perf_counter() - t0

    st = ray.get(agent.stats.remote())
    ray.kill(agent, no_restart=True)

    return {
        "state_kb": state_kb,
        "interval": interval,
        "calls": calls,
        "work_s": work_s,
        "wall_s": wall,
        "wall_per_call_s": wall / calls,
        "t_work_s": st["t_work"],
        "t_ckpt_s": st["t_ckpt"],
        "n_ckpt": st["n_ckpt"],
        "ckpt_bytes": st["ckpt_bytes"],
        "delta_per_ckpt_s": (st["t_ckpt"] / st["n_ckpt"]) if st["n_ckpt"] else 0.0,
        "rpc_s": rpc_s,
        "ckpt_latency_s": ckpt_latency_s,
        "ckpt_mbps": ckpt_mbps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1200)
    ap.add_argument("--work-ms", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument(
        "--ckpt-latency-ms",
        type=float,
        default=0.0,
        help="latência fixa por checkpoint, em ms (0 = disco local)",
    )
    ap.add_argument(
        "--ckpt-mbps",
        type=float,
        default=0.0,
        help="vazão do storage de checkpoint, em MB/s (0 = disco local)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="overhead.csv",
        help="nome do CSV em results/ (use outro nome para não sobrescrever a corrida principal)",
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    intervals, sizes = INTERVALS, STATE_SIZES_KB
    if args.quick:
        args.calls, args.reps = 300, 1
        intervals = [1, 5, 20, 100, 0]
        sizes = [512]

    init_ray()
    rows: list[dict] = []
    total = len(sizes) * len(intervals) * args.reps
    done = 0
    for state_kb in sizes:
        for interval in intervals:
            for rep in range(args.reps):
                r = run_one(
                    interval=interval,
                    state_kb=state_kb,
                    calls=args.calls,
                    work_s=args.work_ms / 1000.0,
                    warmup=args.warmup,
                    tag=f"{state_kb}-{interval}-{rep}",
                    ckpt_latency_s=args.ckpt_latency_ms / 1000.0,
                    ckpt_mbps=args.ckpt_mbps,
                )
                r["rep"] = rep
                rows.append(r)
                done += 1
                label = "sem ckpt" if interval == 0 else f"N={interval}"
                print(
                    f"[{done:>3}/{total}] estado={state_kb:>5}KiB {label:>10} "
                    f"rep={rep}  {r['wall_per_call_s'] * 1e3:7.3f} ms/chamada  "
                    f"delta={r['delta_per_ckpt_s'] * 1e3:6.3f} ms"
                )
    write_csv(RESULTS / args.out, rows)
    ray.shutdown()


if __name__ == "__main__":
    main()
