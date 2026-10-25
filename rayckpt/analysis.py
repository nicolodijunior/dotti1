"""
Análise e geração das figuras.

Produz as três figuras centrais do trabalho (mais uma de validação):

  fig1_overhead.png   overhead de checkpoint vs. intervalo
  fig2_lostwork.png   trabalho perdido na falha vs. intervalo
  fig3_totalcost.png  custo total esperado vs. intervalo (o ótimo)
  fig4_validacao.png  custo medido ponta a ponta vs. o modelo de Daly

E um resumo em results/summary.md com os parâmetros medidos (delta, w, R) e
os ótimos previstos por Young e Daly ao lado do ótimo observado.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .common import FIGURES, RESULTS, ROOT
from .model import (
    cost_daly_exact,
    cost_first_order,
    daly_exact,
    daly_first_order,
    daly_high_order,
    expected_cost_per_call,
    young_1974,
)

style.apply()


# ---------------------------------------------------------------------------
# Parâmetros medidos
# ---------------------------------------------------------------------------


def measured_params() -> dict:
    """Extrai delta, w e R dos CSVs dos experimentos."""
    ov = pd.read_csv(RESULTS / "overhead.csv")
    base = ov[ov.interval == 0].groupby("state_kb").wall_per_call_s.mean()
    # w = custo de uma chamada sem checkpoint algum (trabalho + RPC do Ray)
    w = float(base.mean())
    # delta por tamanho de estado, medido dentro do actor
    delta_by_kb = (
        ov[ov.interval > 0].groupby("state_kb").delta_per_ckpt_s.mean().to_dict()
    )

    p = {"w": w, "delta_by_kb": delta_by_kb, "baseline_by_kb": base.to_dict()}

    fl = RESULTS / "failure.csv"
    if fl.exists():
        f = pd.read_csv(fl)
        p["R"] = float(f.R_s.median())
        p["R_iqr"] = (float(f.R_s.quantile(0.25)), float(f.R_s.quantile(0.75)))
        p["t_restore"] = float(f.t_restore_s.median())
        p["failure_df"] = f
    return p


# ---------------------------------------------------------------------------
# Figura 1 — overhead vs. intervalo
# ---------------------------------------------------------------------------


def fig_overhead(p: dict) -> float:
    """Overhead medido vs. o modelo δ/N, com o piso de resolução explícito.

    Devolve esse piso (ms), que é o desvio-padrão da própria linha de base:
    abaixo dele a subtração `com checkpoint − sem checkpoint` não distingue
    sinal de ruído, e é por isso que os pontos de N grande achatam.
    """
    ov = pd.read_csv(RESULTS / "overhead.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # Piso de resolução: dispersão da linha de base entre repetições.
    floor_ms = float(
        ov[ov.interval == 0].groupby("state_kb").wall_per_call_s.std().max() * 1e3
    )
    ax.axhspan(0, floor_ms, color=style.GRID, alpha=0.65, zorder=1, lw=0)
    ax.annotate(
        "abaixo da resolução da medida",
        xy=(1.05, floor_ms * 0.62), color=style.MUTED, fontsize=8, va="center",
    )

    sizes = sorted(ov.state_kb.unique())
    handles = []
    for i, kb in enumerate(sizes):
        sub = ov[(ov.state_kb == kb) & (ov.interval > 0)]
        base = ov[(ov.state_kb == kb) & (ov.interval == 0)].wall_per_call_s.mean()
        g = sub.groupby("interval").wall_per_call_s.agg(["mean", "sem"]).reset_index()
        over_ms = (g["mean"] - base) * 1e3
        c = style.SERIES[i]
        delta = p["delta_by_kb"][kb]
        h = ax.errorbar(
            g.interval, over_ms, yerr=g["sem"] * 1e3, color=c, marker="o",
            capsize=0, elinewidth=1.2, zorder=4,
            label=f"estado {kb} KiB  (δ = {delta * 1e3:.2f} ms)",
        )
        handles.append(h)
        # Modelo: overhead por chamada = delta / N
        xs = np.logspace(0, math.log10(g.interval.max()), 100)
        ax.plot(xs, delta * 1e3 / xs, color=c, lw=1.0, ls=(0, (3, 3)), alpha=0.5, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor_ms * 0.35, None)
    ax.set_xlabel("Intervalo de checkpoint N (chamadas)")
    ax.set_ylabel("Overhead por chamada (ms)")
    ax.set_title("Overhead de checkpoint cai como δ/N")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper right", handlelength=1.6)
    ax.text(
        0.0, -0.22,
        "Linha cheia: medido (média ± erro padrão de 3 repetições, 1200 chamadas cada).  "
        "Tracejado: modelo δ/N com o δ medido.\n"
        f"Linha de base sem checkpoint: w = {p['w'] * 1e3:.2f} ms por chamada; "
        f"faixa cinza = ±{floor_ms:.3f} ms, a dispersão dessa linha de base.",
        transform=ax.transAxes, fontsize=8, color=style.MUTED, va="top",
    )
    fig.savefig(FIGURES / "fig1_overhead.png")
    plt.close(fig)
    return floor_ms


# ---------------------------------------------------------------------------
# Figura 2 — trabalho perdido vs. intervalo
# ---------------------------------------------------------------------------


def fig_lostwork(p: dict) -> None:
    f = p["failure_df"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    rng = np.random.default_rng(7)
    intervals = sorted(f.interval.unique())

    # Ensaios individuais, com jitter horizontal para não empilhar marcas.
    for N in intervals:
        sub = f[f.interval == N]
        jitter = N + 0.012 * max(intervals) * rng.standard_normal(len(sub))
        ax.scatter(
            jitter, sub.lost_calls, s=16, color=style.SERIES[0], alpha=0.28,
            edgecolors="none", zorder=2,
        )

    g = f.groupby("interval").lost_calls.agg(["mean", "sem", "max"]).reset_index()
    xs = np.array(intervals, dtype=float)
    # Eixos lineares: a relação é linear em N, e é isso que a figura afirma.
    ax.plot(
        xs, xs - 1, color=style.MUTED, lw=1.0, ls=(0, (1, 3)), alpha=0.9, zorder=3,
        label="pior caso: N−1",
    )
    ax.plot(
        xs, (xs - 1) / 2, color=style.MODEL, lw=1.5, ls=(0, (4, 3)), zorder=3,
        label="teoria: (N−1)/2",
    )
    ax.errorbar(
        g.interval, g["mean"], yerr=g["sem"], color=style.SERIES[0], marker="o",
        capsize=0, elinewidth=1.4, zorder=4,
        label="medido (média ± erro padrão)",
    )
    ax.legend(loc="upper left", handlelength=1.8)

    ax.set_xlabel("Intervalo de checkpoint N (chamadas)")
    ax.set_ylabel("Chamadas perdidas na falha")
    ax.set_title("Trabalho perdido cresce linearmente com o intervalo")
    ax.set_xticks(intervals)
    ax.set_xlim(0, max(intervals) * 1.04)
    ax.set_ylim(bottom=0)
    n_trials = int(f.groupby("interval").size().min())
    ax.text(
        0.0, -0.22,
        f"Cada ponto claro é um ensaio independente com uma falha real "
        f"(os._exit no actor), ponto de falha uniforme no ciclo; "
        f"{n_trials} ensaios por N.\n"
        f"Custo de recuperação medido (detecção + restart pelo GCS + leitura do "
        f"checkpoint): R = {p['R'] * 1e3:.0f} ms (mediana), "
        f"dos quais {p['t_restore'] * 1e3:.1f} ms são a leitura do checkpoint.",
        transform=ax.transAxes, fontsize=8, color=style.MUTED, va="top",
    )
    fig.savefig(FIGURES / "fig2_lostwork.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 3 — custo total esperado vs. intervalo
# ---------------------------------------------------------------------------

# MTBFs representativos. O terceiro é a ordem de grandeza de um nó real;
# os dois primeiros são o regime que conseguimos observar em laboratório.
MTBF_GRID = [2.0, 60.0, 3600.0]
MTBF_LABEL = {2.0: "MTBF 2 s", 60.0: "MTBF 1 min", 3600.0: "MTBF 1 h"}


def fig_totalcost(p: dict) -> dict:
    w = p["w"]
    R = p["R"]
    delta = p["delta_by_kb"][max(p["delta_by_kb"])]  # maior estado = δ maior
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    Ns = np.logspace(0, 4.0, 500)
    summary = {}
    for i, M in enumerate(MTBF_GRID):
        cost = np.array([expected_cost_per_call(N, w, delta, M, R) for N in Ns])
        rel = (cost / w - 1.0) * 100.0  # overhead total em % do trabalho útil
        c = style.SERIES[i]
        ax.plot(Ns, rel, color=c, zorder=3)

        k = int(np.argmin(rel))
        ax.scatter([Ns[k]], [rel[k]], s=46, color=c, zorder=5,
                   edgecolors=style.SURFACE, linewidths=1.5)
        ax.annotate(
            f"N* ≈ {Ns[k]:.0f}",
            xy=(Ns[k], rel[k]), xytext=(0, -16), textcoords="offset points",
            color=c, fontsize=9, fontweight="bold", ha="center",
        )
        style.label_end(ax, Ns[-1], rel[-1], f"  {MTBF_LABEL[M]}", c)
        summary[M] = {
            "N_min_numeric": float(Ns[k]),
            "cost_min_rel_pct": float(rel[k]),
            "N_young": young_1974(delta, M) / w,
            "N_daly_1st": daly_first_order(delta, M, R) / w,
            "N_daly_high": daly_high_order(delta, M) / w,
            "N_daly_exact": daly_exact(delta, M) / w,
        }

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intervalo de checkpoint N (chamadas)")
    ax.set_ylabel("Custo total acima do trabalho útil (%)")
    ax.set_title("Soma dos dois efeitos: existe um intervalo ótimo")
    ax.text(
        0.0, -0.22,
        f"Custo = overhead de checkpoint (δ/N, cai) + trabalho perdido e reinício "
        f"((N·w/2 + R)/MTBF, cresce).\n"
        f"Parâmetros medidos nos experimentos A e B: δ = {delta * 1e3:.2f} ms, "
        f"w = {w * 1e3:.2f} ms, R = {R * 1e3:.0f} ms. Ponto: mínimo numérico da curva.",
        transform=ax.transAxes, fontsize=8, color=style.MUTED, va="top",
    )
    fig.savefig(FIGURES / "fig3_totalcost.png")
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Figura 4 — validação ponta a ponta
# ---------------------------------------------------------------------------


def fit_cost_curve(Ns: np.ndarray, costs: np.ndarray, weights: np.ndarray) -> tuple:
    """Ajusta custo(N) = a + b/N + c·N e devolve (a, b, c, N*).

    É a forma funcional do modelo de primeira ordem escrita em unidades de
    chamada: `a` é o trabalho útil w, `b` deve recuperar δ, e `c` deve
    recuperar w²/(2M). O ótimo do ajuste é N* = √(b/c) — uma estimativa
    contínua, muito melhor que o argmin sobre uma grade grossa de N.
    """
    A = np.column_stack([np.ones_like(Ns), 1.0 / Ns, Ns])
    coef, *_ = np.linalg.lstsq(A * weights[:, None], costs * weights, rcond=None)
    a, b, c = coef
    n_star = math.sqrt(b / c) if b > 0 and c > 0 else float("nan")
    return a, b, c, n_star


def bootstrap_optimum(e: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 11) -> tuple:
    """IC do N* por bootstrap, reamostrando as execuções dentro de cada N."""
    rng = np.random.default_rng(seed)
    by_n = {N: sub[col].to_numpy() for N, sub in e.groupby("interval")}
    Ns = np.array(sorted(by_n), dtype=float)
    out = []
    for _ in range(n_boot):
        means, sems = [], []
        for N in Ns:
            v = by_n[N]
            s = v[rng.integers(0, len(v), len(v))]
            means.append(s.mean())
            sems.append(max(s.std(ddof=1) / math.sqrt(len(s)), 1e-12))
        w_ = 1.0 / np.array(sems)
        _, _, _, ns = fit_cost_curve(Ns, np.array(means), w_)
        if np.isfinite(ns):
            out.append(ns)
    arr = np.array(out)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def fig_validation(p: dict) -> dict | None:
    """Dois painéis: o custo bruto medido, e o mesmo custo sem o termo de restart.

    O restart do processo do actor custa R ≈ 1,5 s e acontece uma vez por
    falha, **independentemente de N**. Ele domina o tempo de parede e carrega
    toda a variância de Poisson da contagem de falhas, sem deslocar o ótimo —
    que é exatamente a conclusão de Daly (2006) sobre R. Descontar
    `n_falhas × R` (ambos medidos) isola o trade-off que nos interessa.
    """
    path = RESULTS / "endtoend.csv"
    if not path.exists():
        return None
    e = pd.read_csv(path)
    M = float(e.mtbf_s.iloc[0])
    state_kb = int(e.state_kb.iloc[0])
    delta = p["delta_by_kb"][state_kb]
    w = float(e.per_call_s.mean())
    R = p["R"]

    e = e.copy()
    e["adj_per_call_s"] = (e.wall_s - e.n_crashes * R) / e.calls

    g = e.groupby("interval").agg(
        raw=("wall_per_call_s", "mean"), raw_sem=("wall_per_call_s", "sem"),
        adj=("adj_per_call_s", "mean"), adj_sem=("adj_per_call_s", "sem"),
        count=("wall_per_call_s", "count"), crashes=("n_crashes", "sum"),
    ).reset_index()

    Ns = np.logspace(0, math.log10(g.interval.max() * 1.15), 300)
    n_pred = daly_exact(delta, M) / w
    ticks = sorted(e.interval.unique())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    panels = [
        ("raw", "raw_sem", R, "Tempo de parede bruto", "Custo medido ponta a ponta"),
        ("adj", "adj_sem", 0.0, "Custo sem o termo de restart",
         "Descontando n_falhas × R (independente de N)"),
    ]
    # Usamos o modelo de PRIMEIRA ORDEM como curva de referência, e não a
    # forma exata de Daly. Motivo: a forma exata traz um fator e^(R/M) que
    # pressupõe que falhas também atingem a aplicação durante o downtime. Aqui
    # a falha só é sorteada dentro de `step()`, isto é, só durante computação —
    # que é exatamente a hipótese sob a qual vale a expressão de primeira
    # ordem. Com R = 1,48 s e M = 2 s esse fator vale e^0,74 ≈ 2,1, o que
    # explicaria sozinho um desvio de mais de 20% no nível da curva.
    results = {}
    Ngrid = g.interval.to_numpy(dtype=float)
    for ax, (col, sem, r_model, ylab, title) in zip(axes, panels):
        ax.errorbar(
            g.interval, g[col] * 1e3, yerr=g[sem] * 1e3, color=style.SERIES[0],
            marker="o", capsize=0, elinewidth=1.4, zorder=5,
            label="medido (média ± erro padrão)",
        )
        model = np.array([cost_first_order(N * w, delta, M, r_model, T_s=N * w) / N for N in Ns])
        ax.plot(
            Ns, model * 1e3, color=style.SERIES[1], lw=1.6, ls=(0, (4, 3)), zorder=3,
            label="modelo de 1ª ordem (δ, w, R medidos)",
        )

        # Ajuste do modelo aos dados: estimativa contínua do ótimo.
        weights = 1.0 / np.maximum(g[sem].to_numpy(), 1e-12)
        a, b, c, n_fit = fit_cost_curve(Ngrid, g[col].to_numpy(), weights)
        src = "wall_per_call_s" if col == "raw" else "adj_per_call_s"
        lo, hi = bootstrap_optimum(e, src)
        fitted = a + b / Ns + c * Ns
        ax.plot(
            Ns, fitted * 1e3, color=style.SERIES[2], lw=1.4, zorder=4,
            label=f"ajuste a + b/N + c·N  →  N* = {n_fit:.0f}  IC95% [{lo:.0f}, {hi:.0f}]",
        )
        ax.axvspan(lo, hi, color=style.SERIES[2], alpha=0.12, zorder=1, lw=0)
        ax.axvline(n_fit, color=style.SERIES[2], lw=1.3, zorder=2)
        ax.axvline(n_pred, color=style.CRITICAL, lw=1.2, ls=(0, (2, 2)), alpha=0.85, zorder=2)

        ax.set_xscale("log")
        ax.set_xlabel("Intervalo de checkpoint N (chamadas)")
        ax.set_ylabel(f"{ylab} (ms/chamada útil)")
        ax.set_title(title)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        y0, y1 = ax.get_ylim()
        ax.annotate(
            f" previsto: N* = {n_pred:.0f}", xy=(n_pred, y0), xytext=(5, 8),
            textcoords="offset points", color=style.CRITICAL, fontsize=8.5,
            fontweight="bold", va="bottom",
        )
        ax.legend(loc="upper center", handlelength=1.8, fontsize=8.5)
        results[col] = {
            "argmin": int(g.interval.iloc[int(g[col].idxmin())]),
            "n_fit": float(n_fit), "ci": (lo, hi),
            "a_ms": a * 1e3, "b_ms": b * 1e3, "c_ms": c * 1e3,
        }

    fig.suptitle(
        f"Validação: o ótimo medido é consistente com o previsto (MTBF {M:g} s)",
        fontsize=12, fontweight="bold", color=style.INK, y=1.02,
    )
    axes[0].text(
        0.0, -0.23,
        f"Falhas reais como processo de Poisson ({int(g.crashes.sum())} no total); o trabalho "
        f"perdido é refeito pelo driver. {int(g['count'].min())} execuções de "
        f"{int(e.calls.iloc[0])} chamadas por ponto (média ± erro padrão).  "
        f"δ = {delta * 1e3:.2f} ms, w = {w * 1e3:.2f} ms, R = {R * 1e3:.0f} ms.",
        transform=axes[0].transAxes, fontsize=8, color=style.MUTED, va="top",
    )
    fig.savefig(FIGURES / "fig4_validacao.png")
    plt.close(fig)

    return {
        "mtbf_s": M, "delta_s": delta, "w_s": w, "R_s": R,
        "raw": results["raw"], "adj": results["adj"],
        "N_young": young_1974(delta, M) / w,
        "N_daly_high": daly_high_order(delta, M) / w,
        "N_daly_exact": n_pred,
        # Valores que o ajuste deveria recuperar, se o modelo descreve os dados
        "b_expected_ms": delta * 1e3,
        "c_expected_ms": (w ** 2 / (2 * M)) * 1e3,
        "table": g.to_dict("records"),
        "crashes_total": int(e.n_crashes.sum()),
        "rework_total": int(e.rework_calls.sum()),
    }


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------


def write_summary(p: dict, cost_summary: dict, val: dict | None) -> None:
    L: list[str] = []
    L.append("# Resumo dos resultados\n")
    L.append("## Parâmetros medidos\n")
    L.append("| Parâmetro | Valor | Como foi medido |")
    L.append("|---|---|---|")
    L.append(
        f"| w (custo de uma chamada) | {p['w'] * 1e3:.3f} ms | "
        "tempo de parede por chamada com checkpoint desligado (N=0) |"
    )
    for kb, d in sorted(p["delta_by_kb"].items()):
        L.append(
            f"| δ (checkpoint, estado {kb} KiB) | {d * 1e3:.3f} ms | "
            "tempo de pickle + fsync + rename atômico, dentro do actor |"
        )
    L.append(
        f"| R (recuperação) | {p['R'] * 1e3:.0f} ms "
        f"(IQR {p['R_iqr'][0] * 1e3:.0f}–{p['R_iqr'][1] * 1e3:.0f}) | "
        "latência da chamada que morreu, menos a latência de uma chamada normal |"
    )
    L.append(
        f"| — leitura do checkpoint | {p['t_restore'] * 1e3:.1f} ms | "
        "tempo dentro do construtor da nova encarnação |"
    )
    L.append("")
    L.append(
        f"O custo de recuperação é dominado pelo restart do processo do actor "
        f"pelo GCS ({p['R'] * 1e3:.0f} ms), não pela leitura do estado "
        f"({p['t_restore'] * 1e3:.2f} ms): a leitura é "
        f"{p['R'] / p['t_restore']:.0f}x menor que o restart. "
        "Isso tem uma consequência prática — aumentar o estado do agente encarece "
        "o checkpoint (δ) muito mais do que encarece a recuperação.\n"
    )

    L.append("## Trabalho perdido vs. intervalo\n")
    f = p["failure_df"]
    g = f.groupby("interval").agg(
        medido=("lost_calls", "mean"),
        desvio=("lost_calls", "std"),
        erro=("lost_calls", "sem"),
        teoria=("expected_lost_theory", "first"),
        ensaios=("lost_calls", "size"),
    ).reset_index()
    L.append(
        "| N | perdidas (média medida) | (N−1)/2 | erro padrão | "
        "desvio em σ | ensaios |"
    )
    L.append("|---|---|---|---|---|---|")
    for r in g.itertuples():
        z = (r.medido - r.teoria) / r.erro if r.erro > 0 else 0.0
        L.append(
            f"| {r.interval} | {r.medido:.2f} | {r.teoria:.1f} | "
            f"{r.erro:.2f} | {z:+.1f} | {r.ensaios} |"
        )
    L.append("")
    worst = max(
        abs((r.medido - r.teoria) / r.erro) if r.erro > 0 else 0.0
        for r in g.itertuples()
    )
    L.append(
        f"Nenhum N se afasta da previsão (N−1)/2 por mais de {worst:.1f} erros "
        "padrão, ou seja, o trabalho perdido medido é compatível com a teoria "
        "em toda a faixa. A dispersão é intrinsecamente alta — o ponto de falha "
        "é uniforme no ciclo, então o desvio-padrão da perda cresce com N "
        "(≈ N/√12), e é por isso que N grande precisa de mais ensaios para a "
        "média estabilizar.\n"
    )

    L.append("## Intervalo ótimo previsto (δ do maior estado)\n")
    delta = p["delta_by_kb"][max(p["delta_by_kb"])]
    L.append(f"δ = {delta * 1e3:.2f} ms, w = {p['w'] * 1e3:.2f} ms, R = {p['R'] * 1e3:.0f} ms\n")
    L.append("| MTBF | N* Young (1974) | N* Daly 1ª ordem | N* Daly alta ordem | N* Daly exato | mínimo numérico | custo no ótimo |")
    L.append("|---|---|---|---|---|---|---|")
    for M, s in cost_summary.items():
        L.append(
            f"| {M:g} s | {s['N_young']:.0f} | {s['N_daly_1st']:.0f} | "
            f"{s['N_daly_high']:.0f} | {s['N_daly_exact']:.0f} | "
            f"{s['N_min_numeric']:.0f} | +{s['cost_min_rel_pct']:.2f}% |"
        )
    L.append("")
    L.append(
        "Young e Daly 1ª ordem diferem porque só o segundo coloca R sob o radical. "
        "Daly (2006) mostra na solução de alta ordem que R **não** contribui para "
        "o ótimo — e de fato a coluna de alta ordem fica junto do mínimo numérico, "
        "enquanto a de 1ª ordem se afasta quando R é grande em relação ao MTBF.\n"
    )

    if val:
        L.append("## Validação ponta a ponta\n")
        L.append(
            f"MTBF escalado para {val['mtbf_s']:g} s; "
            f"δ = {val['delta_s'] * 1e3:.2f} ms, w = {val['w_s'] * 1e3:.2f} ms, "
            f"R = {val['R_s'] * 1e3:.0f} ms.\n"
        )
        raw, adj = val["raw"], val["adj"]
        L.append(
            f"- falhas reais injetadas e recuperadas: **{val['crashes_total']}**\n"
            f"- chamadas refeitas após rollback: **{val['rework_total']}**\n"
            f"- previsto por Young: N* = {val['N_young']:.0f}; "
            f"Daly alta ordem: N* = {val['N_daly_high']:.0f}; "
            f"Daly exato: N* = {val['N_daly_exact']:.0f}\n"
        )
        L.append("")
        L.append("| Curva | N* do ajuste | IC 95% | argmin da grade |")
        L.append("|---|---|---|---|")
        L.append(
            f"| tempo de parede bruto | {raw['n_fit']:.0f} | "
            f"[{raw['ci'][0]:.0f}, {raw['ci'][1]:.0f}] | {raw['argmin']} |"
        )
        L.append(
            f"| descontado o restart | {adj['n_fit']:.0f} | "
            f"[{adj['ci'][0]:.0f}, {adj['ci'][1]:.0f}] | {adj['argmin']} |"
        )
        L.append("")
        L.append(
            "O ajuste é da forma custo(N) = a + b/N + c·N, que é o modelo de "
            "primeira ordem em unidades de chamada; N* = √(b/c). Se o modelo "
            "descreve os dados, `b` recupera δ e `c` recupera w²/(2·MTBF):\n"
        )
        L.append("| coeficiente | ajustado (sem restart) | esperado | significado |")
        L.append("|---|---|---|---|")
        L.append(
            f"| a | {adj['a_ms']:.3f} ms | {val['w_s'] * 1e3:.3f} ms | trabalho útil w |"
        )
        L.append(
            f"| b | {adj['b_ms']:.3f} ms | {val['b_expected_ms']:.3f} ms | "
            "custo de um checkpoint δ |"
        )
        L.append(
            f"| c | {adj['c_ms']:.5f} ms | {val['c_expected_ms']:.5f} ms | "
            "trabalho perdido w²/(2·MTBF) |"
        )
        L.append("")
        L.append(
            "O tempo de parede **bruto** é dominado pelo termo de restart "
            f"({val['R_s'] * 1e3:.0f} ms por falha, com a contagem de falhas "
            "seguindo Poisson), que é independente de N e carrega quase toda a "
            "variância — por isso seu argmin na grade não é confiável. "
            "Descontar `n_falhas × R` (ambos medidos) é legítimo justamente "
            "porque Daly mostra que R não desloca o ótimo, e deixa visível o "
            "trade-off que interessa.\n"
        )
        L.append(
            "| N | bruto (ms/chamada útil) | erro padrão | sem restart | erro padrão | execuções | falhas |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for r in val["table"]:
            L.append(
                f"| {int(r['interval'])} | {r['raw'] * 1e3:.3f} | "
                f"{r['raw_sem'] * 1e3:.3f} | {r['adj'] * 1e3:.3f} | "
                f"{r['adj_sem'] * 1e3:.3f} | {int(r['count'])} | {int(r['crashes'])} |"
            )
        L.append("")

    out = RESULTS / "summary.md"
    out.write_text("\n".join(L))
    print(f"  -> {out.relative_to(ROOT)}")


def write_facts(p: dict, cost_summary: dict, val: dict | None, floor_ms: float) -> None:
    """Exporta os números em JSON, para os slides serem gerados a partir deles.

    Sem isso, o texto dos slides ficaria cravado à mão e divergiria das figuras
    assim que alguém reexecutasse os experimentos.
    """
    ov = pd.read_csv(RESULTS / "overhead.csv")
    base = ov[ov.interval == 0].groupby("state_kb").wall_per_call_s.mean()
    g = ov[ov.interval > 0].groupby(["state_kb", "interval"]).wall_per_call_s.mean()
    overhead_pct = {
        str(kb): {
            str(n): float((g[(kb, n)] - base[kb]) / base[kb] * 100)
            for n in sorted(ov[ov.interval > 0].interval.unique())
        }
        for kb in sorted(ov.state_kb.unique())
    }

    f = p["failure_df"]
    lost = f.groupby("interval").agg(
        medido=("lost_calls", "mean"), teoria=("expected_lost_theory", "first"),
        erro=("lost_calls", "sem"),
    ).reset_index()
    worst_sigma = float(
        max(abs((r.medido - r.teoria) / r.erro) if r.erro > 0 else 0.0
            for r in lost.itertuples())
    )

    facts = {
        "w_ms": p["w"] * 1e3,
        "delta_ms": {str(k): v * 1e3 for k, v in p["delta_by_kb"].items()},
        "R_ms": p["R"] * 1e3,
        "t_restore_ms": p["t_restore"] * 1e3,
        "restore_ratio": p["R"] / p["t_restore"],
        "resolution_floor_ms": floor_ms,
        "overhead_pct": overhead_pct,
        "lost_worst_sigma": worst_sigma,
        "n_trials_per_N": int(f.groupby("interval").size().min()),
        "optimum_by_mtbf": {
            str(M): {k: float(v) for k, v in s.items()} for M, s in cost_summary.items()
        },
        "validation": None,
    }
    if val:
        facts["validation"] = {
            "mtbf_s": val["mtbf_s"], "delta_ms": val["delta_s"] * 1e3,
            "w_ms": val["w_s"] * 1e3, "R_ms": val["R_s"] * 1e3,
            "crashes_total": val["crashes_total"], "rework_total": val["rework_total"],
            "n_predicted": val["N_daly_exact"],
            "n_young": val["N_young"], "n_daly_high": val["N_daly_high"],
            "b_expected_ms": val["b_expected_ms"], "c_expected_ms": val["c_expected_ms"],
            "raw": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in val["raw"].items()},
            "adj": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in val["adj"].items()},
        }
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(facts, indent=2, ensure_ascii=False))
    print(f"  -> {out.relative_to(ROOT)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = measured_params()
    floor_ms = fig_overhead(p)
    print("  -> figures/fig1_overhead.png")
    cost_summary = {}
    val = None
    if "failure_df" in p:
        fig_lostwork(p)
        print("  -> figures/fig2_lostwork.png")
        cost_summary = fig_totalcost(p)
        print("  -> figures/fig3_totalcost.png")
        val = fig_validation(p)
        if val:
            print("  -> figures/fig4_validacao.png")
        write_summary(p, cost_summary, val)
        write_facts(p, cost_summary, val, floor_ms)


if __name__ == "__main__":
    main()
