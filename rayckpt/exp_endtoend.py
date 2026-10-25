"""
Experimento C/D — Custo total medido ponta a ponta vs. intervalo de checkpoint.

Pergunta: somando overhead de checkpoint e trabalho perdido, existe mesmo um
intervalo ótimo (um mínimo), como prevê Young (1974)?

Diferença em relação ao Experimento B: aqui as falhas são um processo de
Poisson ao longo de toda a execução (não uma única falha injetada), e o
trabalho perdido é **refeito**. Esse é o ponto importante para comparar com
Young/Daly: o modelo deles conta o rework no tempo de parede, então o driver
precisa de fato voltar ao ponto do último checkpoint e reexecutar as chamadas
perdidas.

Como o laço do driver é guiado pelo `calls_done` que o actor devolve (e não
por um contador do driver), o rewind acontece naturalmente: depois de um
restart o actor responde com o contador restaurado do checkpoint, e o driver
simplesmente continua dali, reexecutando o que se perdeu.

O MTBF é escalado para a casa dos segundos. Um cluster real falha em horas ou
dias, o que tornaria o experimento inviável; escalando MTBF e δ juntos
preservamos a razão adimensional xi = sqrt(delta/2M) que governa as fórmulas,
que é o que precisamos para testar o modelo.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import ray

from .agent import ToolCallingAgent
from .common import RESULTS, fresh_workdir, init_ray, measure_rpc_latency, write_csv

INTERVALS = [1, 5, 10, 25, 50, 100, 200, 400]


def run_one(
    interval: int,
    run: int,
    calls: int,
    work_s: float,
    state_kb: int,
    mtbf_s: float,
    warmup: int,
    tag: str,
) -> dict:
    """Executa `calls` chamadas úteis sob falhas de Poisson e mede o custo."""
    workdir = fresh_workdir(f"e2e-{tag}")
    agent = ToolCallingAgent.remote(
        workdir=str(workdir),
        checkpoint_every=interval,
        work_s=work_s,
        state_kb=state_kb,
    )

    # Aquecimento sem falhas: paga o custo de criação do actor e do primeiro
    # toque no disco fora da janela medida.
    for i in range(warmup):
        ray.get(agent.step.remote(i + 1))
    base_done = ray.get(agent.stats.remote())["calls_done"]
    per_call_s = work_s + measure_rpc_latency(agent, n=60)
    # Probabilidade de falha por chamada que produz o MTBF pedido.
    p_crash = min(0.5, per_call_s / mtbf_s)

    cur_pid = ray.get(agent.probe.remote())
    target = base_done + calls

    last_done = base_done
    executions = 0
    n_crashes = 0
    lost_total = 0
    recovery_total_s = 0.0

    t0 = time.perf_counter()
    while last_done < target:
        nxt = last_done + 1
        ts = time.perf_counter()
        res = ray.get(agent.step.remote(nxt, None, p_crash))
        dt = time.perf_counter() - ts
        executions += 1

        if res["pid"] != cur_pid:
            # Nasceu uma nova encarnação: houve falha + restart + restore.
            n_crashes += 1
            lost_total += last_done - res["calls_at_birth"]
            recovery_total_s += dt
            cur_pid = res["pid"]

        last_done = res["calls_done"]
    wall = time.perf_counter() - t0

    st = ray.get(agent.stats.remote())
    ray.kill(agent, no_restart=True)

    # Decomposição do custo. `t_ckpt` e `t_work` zeram a cada restart, por
    # isso o tempo de parede do driver é a medida de referência; a
    # decomposição usa as contagens, que são exatas.
    rework_s = lost_total * per_call_s

    return {
        "interval": interval,
        "run": run,
        "calls": calls,
        "executions": executions,
        "rework_calls": lost_total,
        "n_crashes": n_crashes,
        "mtbf_s": mtbf_s,
        "p_crash": p_crash,
        "per_call_s": per_call_s,
        "work_s": work_s,
        "state_kb": state_kb,
        "wall_s": wall,
        "wall_per_call_s": wall / calls,
        "recovery_total_s": recovery_total_s,
        "rework_s": rework_s,
        "n_ckpt_last_incarnation": st["n_ckpt"],
        "ckpt_bytes": st["ckpt_bytes"],
        "final_calls_done": st["calls_done"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1200)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--work-ms", type=float, default=2.0)
    ap.add_argument("--state-kb", type=int, default=4096)
    ap.add_argument("--mtbf-s", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    intervals = INTERVALS
    if args.quick:
        args.calls, args.runs = 300, 2
        intervals = [1, 25, 200]

    init_ray()
    rows: list[dict] = []
    total = len(intervals) * args.runs
    done = 0
    for interval in intervals:
        for run in range(args.runs):
            r = run_one(
                interval=interval,
                run=run,
                calls=args.calls,
                work_s=args.work_ms / 1000.0,
                state_kb=args.state_kb,
                mtbf_s=args.mtbf_s,
                warmup=args.warmup,
                tag=f"{interval}-{run}",
            )
            rows.append(r)
            done += 1
            print(
                f"[{done:>3}/{total}] N={interval:>3} r={run:>2}  "
                f"wall={r['wall_s']:6.2f}s  {r['wall_per_call_s'] * 1e3:6.2f} ms/chamada  "
                f"falhas={r['n_crashes']:>2}  refeitas={r['rework_calls']:>4}"
            )
    write_csv(RESULTS / "endtoend.csv", rows)
    ray.shutdown()


if __name__ == "__main__":
    main()
