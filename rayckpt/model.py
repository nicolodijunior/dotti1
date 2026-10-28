"""
Modelos analíticos de intervalo ótimo de checkpoint.

Notação (a de Daly 2006, para podermos comparar número a número):

    tau   intervalo de computação entre dois checkpoints [s]
    delta custo de gravar um checkpoint [s]            (δ)
    M     tempo médio entre falhas / MTTI [s]
    R     custo de reinício + restauração [s]
    T_s   tempo de solução útil (sem overhead) [s]

Referências:
  Young, J.W. (1974) "A first order approximation to the optimum checkpoint
    interval", CACM 17(9):530-531.          ->  tau_opt = sqrt(2 * delta * M)
  Daly, J.T. (2006) "A higher order estimate of the optimum checkpoint
    interval for restart dumps", FGCS 22(3):303-312.
      eq. (12)  primeira ordem com reinício: tau_opt = sqrt(2*delta*(M+R))
      eq. (25)  solução exata via função de Lambert
      eq. (37)  solução de perturbação de alta ordem (erro < 0,2% em T_w)
"""

from __future__ import annotations

import math

from scipy.special import lambertw

from .agent import storage_delay_s

# ---------------------------------------------------------------------------
# Intervalo ótimo
# ---------------------------------------------------------------------------


def young_1974(delta: float, M: float) -> float:
    """tau_opt = sqrt(2 * delta * M). Young (1974), primeira ordem."""
    return math.sqrt(2.0 * delta * M)


def daly_first_order(delta: float, M: float, R: float = 0.0) -> float:
    """tau_opt = sqrt(2*delta*(M+R)). Daly (2006), eq. (12).

    É o Young com o custo de reinício sob o radical. Daly mostra depois que
    esse termo R desaparece na solução de alta ordem — algo que nosso
    experimento pode checar, porque medimos R diretamente.
    """
    return math.sqrt(2.0 * delta * (M + R))


def daly_high_order(delta: float, M: float) -> float:
    """Daly (2006), eq. (37) — os três primeiros termos da série."""
    if delta >= 2.0 * M:
        return M
    x = delta / (2.0 * M)
    return math.sqrt(2.0 * delta * M) * (1.0 + (1.0 / 3.0) * math.sqrt(x) + (1.0 / 9.0) * x) - delta


def daly_exact(delta: float, M: float) -> float:
    """Daly (2006), eq. (25): solução exata via função W de Lambert.

    Com xi = sqrt(delta / 2M) e eta = (tau + delta) / M, a condição de
    otimalidade vira  e^(-eta) + eta = 2*xi^2 + 1, cuja raiz é
    eta = 2*xi^2 + 1 + W(-e^(-2*xi^2 - 1)) no ramo principal.
    """
    xi2 = delta / (2.0 * M)
    w = lambertw(-math.exp(-2.0 * xi2 - 1.0), k=0).real
    eta = 2.0 * xi2 + 1.0 + w
    return M * eta - delta


# ---------------------------------------------------------------------------
# Funções de custo
# ---------------------------------------------------------------------------


def cost_first_order(tau: float, delta: float, M: float, R: float, T_s: float = 1.0) -> float:
    """Tempo de parede de primeira ordem (Daly, eqs. 9-10).

        T_w = T_s * [ 1 + delta/tau + (tau/2 + R)/M ]

    Os três termos são exatamente as três ideias do trabalho:
      1 -> trabalho útil
      2 -> overhead de checkpoint (cai com tau)
      3 -> trabalho perdido + reinício por falha (cresce com tau)
    """
    return T_s * (1.0 + delta / tau + (tau / 2.0 + R) / M)


def cost_daly_exact(tau: float, delta: float, M: float, R: float, T_s: float = 1.0) -> float:
    """Tempo de parede exato sob falhas de Poisson (Daly, eq. 36).

        T_w = M * e^(R/M) * (e^((tau+delta)/M) - 1) * T_s/tau
    """
    return M * math.exp(R / M) * (math.exp((tau + delta) / M) - 1.0) * (T_s / tau)


# ---------------------------------------------------------------------------
# Tradução para "unidades de chamada" (o que o experimento mede)
# ---------------------------------------------------------------------------


def optimal_interval_calls(delta: float, M: float, w: float, model: str = "daly_exact") -> float:
    """Converte tau_opt [s] para N_opt [chamadas], dado o custo w de uma chamada."""
    fn = {
        "young": lambda: young_1974(delta, M),
        "daly_first_order": lambda: daly_first_order(delta, M),
        "daly_high_order": lambda: daly_high_order(delta, M),
        "daly_exact": lambda: daly_exact(delta, M),
    }[model]
    return fn() / w


def expected_cost_per_call(N: float, w: float, delta: float, M: float, R: float) -> float:
    """Custo esperado por chamada útil, em segundos, para intervalo de N chamadas.

    É `cost_first_order` reescrito em unidades de chamada (tau = N*w),
    normalizado por chamada. É a curva da terceira figura do trabalho.
    """
    tau = N * w
    return w * (1.0 + delta / tau + (tau / 2.0 + R) / M)


# ---------------------------------------------------------------------------
# Projeção para storage realista
# ---------------------------------------------------------------------------

# Perfis de storage remoto, em (latência por operação [s], vazão [MB/s]).
#
# O δ que medimos é o do disco local, que é o piso otimista: um checkpoint de
# actor precisa estar acessível a todo o cluster, e nesse caso δ é dominado
# por rede, não por disco. Os números abaixo são ordens de grandeza
# publicadas para cada classe de storage, não medições nossas — por isso o
# que sai daqui é rotulado como projeção, e não como resultado medido.
STORAGE_PROFILES: dict[str, tuple[float, float]] = {
    "EFS / NFS": (0.010, 250.0),
    "S3 (stream único)": (0.030, 120.0),
    "S3 (cross-region)": (0.120, 60.0),
}


def delta_remote(state_bytes: int, profile: str, delta_local: float = 0.0) -> float:
    """δ projetado para gravar `state_bytes` no storage `profile`.

    `delta_local` é somado porque serializar e tocar o disco continua
    custando: o storage remoto entra como custo *adicional*, que é exatamente
    como o actor o injeta (ver `agent.storage_delay_s`). Passando o δ local
    medido para o mesmo tamanho de estado, a projeção fica consistente com o
    que uma execução com `--ckpt-latency-ms`/`--ckpt-mbps` mediria.
    """
    latency_s, mbps = STORAGE_PROFILES[profile]
    return delta_local + storage_delay_s(state_bytes, latency_s, mbps)


def crossover_bytes(profile: str) -> float:
    """Tamanho de estado em que latência e transferência se igualam.

    Abaixo dele δ é ~constante (dominado pela latência) e a hipótese de
    Young/Daly de que δ não depende do intervalo vale trivialmente; acima,
    δ cresce proporcionalmente ao estado.
    """
    latency_s, mbps = STORAGE_PROFILES[profile]
    return latency_s * mbps * 1e6


def optimum_table(delta: float, M: float, w: float, R: float = 0.0) -> dict:
    """N* pelos quatro modelos, mais o custo relativo no ótimo.

    `cost_rel_pct` é o excesso sobre o tempo de solução puro no ótimo, isto é
    quanto da execução é overhead inevitável (checkpoint + rework) quando o
    intervalo está bem escolhido.
    """
    n_exact = daly_exact(delta, M) / w
    cost_at_opt = expected_cost_per_call(n_exact, w, delta, M, R)
    return {
        "N_young": young_1974(delta, M) / w,
        "N_daly_1st": daly_first_order(delta, M, R) / w,
        "N_daly_high": daly_high_order(delta, M) / w,
        "N_daly_exact": n_exact,
        "cost_rel_pct": (cost_at_opt / w - 1.0) * 100.0,
        "xi": math.sqrt(delta / (2.0 * M)),
    }


def expected_lost_calls(N: int) -> float:
    """Trabalho perdido esperado, em chamadas, se a falha cai em ponto uniforme.

    Se a falha ocorre após a k-ésima chamada concluída, o estado volta para o
    último múltiplo de N, perdendo `k mod N` chamadas. Com k uniforme sobre um
    ciclo, a média é (N-1)/2.
    """
    return (N - 1) / 2.0
