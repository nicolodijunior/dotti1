"""
Experimento B — Trabalho perdido em caso de falha vs. intervalo de checkpoint.

Pergunta: quantas chamadas se perdem, em média, se o actor cair antes do
próximo checkpoint?

Método: para cada intervalo N fazemos `trials` execuções independentes. Em
cada uma, sorteamos um ponto de falha uniforme dentro de um ciclo e matamos o
processo do actor de verdade, com `os._exit(1)` dentro do método — não é
simulação: o raylet detecta a morte do worker, publica o evento no GCS, e o
GCS ressubmete a creation task do actor (`max_restarts=-1`). O construtor
recarrega o último checkpoint e o `max_task_retries=-1` re-executa a chamada
que falhou.

Duas medições saem daí:
  * trabalho perdido = (chamadas que o driver já tinha visto concluídas)
                       - (chamadas presentes no estado restaurado)
    Valor esperado teórico: (N-1)/2.
  * R = custo de recuperação (detecção + restart do actor + leitura do
    checkpoint), isolado subtraindo a latência de uma chamada normal.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import ray

from .agent import ToolCallingAgent
from .common import RESULTS, fresh_workdir, init_ray, write_csv

INTERVALS = [1, 5, 10, 20, 50, 100]


def run_trial(
    interval: int,
    trial: int,
    work_s: float,
    state_kb: int,
    cycles: int,
    rng: random.Random,
    ckpt_latency_s: float = 0.0,
    ckpt_mbps: float = 0.0,
) -> dict:
    total = max(40, cycles * interval)
    # Ponto de falha uniforme, depois de ao menos um ciclo completo de
    # checkpoint (para que exista um checkpoint para onde voltar).
    crash_at = rng.randint(interval + 1, total)

    workdir = fresh_workdir(f"failure-{interval}-{trial}")
    agent = ToolCallingAgent.remote(
        workdir=str(workdir),
        checkpoint_every=interval,
        work_s=work_s,
        state_kb=state_kb,
        ckpt_latency_s=ckpt_latency_s,
        ckpt_mbps=ckpt_mbps,
    )

    latencies: list[float] = []
    recovery_s = float("nan")
    lost = -1
    restored_at = -1

    for i in range(1, total + 1):
        t0 = time.perf_counter()
        res = ray.get(agent.step.remote(i, crash_at if i == crash_at else None))
        dt = time.perf_counter() - t0

        if i == crash_at:
            # Esta chamada derrubou o actor e foi re-executada
            # automaticamente na nova encarnação.
            recovery_s = dt
            restored_at = res["calls_at_birth"]
            # O driver tinha visto crash_at-1 chamadas concluídas.
            lost = (crash_at - 1) - restored_at
        else:
            latencies.append(dt)

    st = ray.get(agent.stats.remote())
    ray.kill(agent, no_restart=True)

    normal_s = statistics.median(latencies)
    return {
        "interval": interval,
        "trial": trial,
        "total_calls": total,
        "crash_at": crash_at,
        "restored_calls_done": restored_at,
        "lost_calls": lost,
        "expected_lost_theory": (interval - 1) / 2.0,
        "recovery_total_s": recovery_s,
        "normal_call_s": normal_s,
        # R = custo de recuperação puro (sem o trabalho da própria chamada).
        "R_s": recovery_s - normal_s,
        "t_restore_s": st["t_restore"],
        "ckpt_bytes": st["ckpt_bytes"],
        "final_calls_done": st["calls_done"],
        "restarted": st["restored"],
        "ckpt_latency_s": ckpt_latency_s,
        "ckpt_mbps": ckpt_mbps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--work-ms", type=float, default=2.0)
    ap.add_argument("--state-kb", type=int, default=512)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261024)
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
        default="failure.csv",
        help="nome do CSV em results/ (use outro nome para não sobrescrever a corrida principal)",
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    intervals = INTERVALS
    if args.quick:
        args.trials = 5
        intervals = [1, 10, 50]

    init_ray()
    rng = random.Random(args.seed)
    rows: list[dict] = []
    total = len(intervals) * args.trials
    done = 0
    for interval in intervals:
        for trial in range(args.trials):
            r = run_trial(
                interval=interval,
                trial=trial,
                work_s=args.work_ms / 1000.0,
                state_kb=args.state_kb,
                ckpt_latency_s=args.ckpt_latency_ms / 1000.0,
                ckpt_mbps=args.ckpt_mbps,
                cycles=args.cycles,
                rng=rng,
            )
            rows.append(r)
            done += 1
            print(
                f"[{done:>3}/{total}] N={interval:>3} t={trial:>2}  "
                f"falha@{r['crash_at']:>4}  perdidas={r['lost_calls']:>3} "
                f"(teoria {r['expected_lost_theory']:.1f})  "
                f"R={r['R_s'] * 1e3:7.1f} ms  restaurou={r['restarted']}"
            )
    write_csv(RESULTS / args.out, rows)
    ray.shutdown()


if __name__ == "__main__":
    main()
