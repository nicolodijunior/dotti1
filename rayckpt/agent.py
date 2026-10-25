"""
Actor Ray que simula um agente executando chamadas de ferramenta (tool calls),
com checkpoint manual periódico.

Implementa o padrão *self-managed checkpointing* que a documentação atual de
Ray Actor Fault Tolerance recomenda: o actor é declarado com `max_restarts` /
`max_task_retries`, grava seu estado em armazenamento externo a cada N chamadas
e **restaura o estado no próprio construtor**, que é o código que o Ray
re-executa quando reinicia um actor morto.

Motivação (ver README): no Ray 2.x a reconstrução por linhagem recupera
*objetos*, mas o whitepaper "Ray v2 Architecture" é explícito de que
"objects created by actor tasks are not reconstructable" e que, ao reiniciar,
o estado do actor "will be recreated by rerunning its constructor". Ou seja,
o estado de um actor só sobrevive a uma falha se a aplicação fizer checkpoint
— e nem o paper de 2018 nem a documentação dizem com que frequência fazê-lo.
É esse intervalo que este módulo permite medir.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import ray

# ---------------------------------------------------------------------------
# Trabalho sintético
# ---------------------------------------------------------------------------

# Bloco de iterações executado entre duas leituras do relógio. Mantém o
# busy-wait preciso sem que a chamada a perf_counter domine o custo.
_SPIN_CHUNK = 64


def busy_work(duration_s: float) -> int:
    """Queima CPU por aproximadamente `duration_s` segundos.

    Usamos CPU de verdade (e não `sleep`) para que o actor concorra por um
    núcleo como um rollout real faria, e para que o custo do checkpoint
    apareça como tempo somado ao caminho crítico.
    """
    if duration_s <= 0:
        return 0
    t0 = time.perf_counter()
    spins = 0
    acc = 0.0
    while True:
        for _ in range(_SPIN_CHUNK):
            acc += 1.000001
        spins += _SPIN_CHUNK
        if time.perf_counter() - t0 >= duration_s:
            break
    return spins


# ---------------------------------------------------------------------------
# Estado do agente
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Estado que precisa sobreviver a uma falha do actor.

    `scratch` é um payload de tamanho fixo que representa a memória de
    trabalho do agente (embeddings em cache, contexto serializado, KV cache
    ...). Ele é o que torna o custo de checkpoint δ aproximadamente constante
    e controlável: variando `state_kb` variamos δ sem mudar nada mais do
    experimento.
    """

    calls_done: int = 0
    tool_results: list = field(default_factory=list)
    scratch: bytes = b""

    def checksum(self) -> int:
        """Invariante simples para provar que o estado restaurado é coerente."""
        return (self.calls_done * 2654435761) ^ len(self.scratch)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
class ToolCallingAgent:
    """Agente que executa uma tool call por chamada e faz checkpoint a cada N.

    Parameters
    ----------
    workdir:
        Diretório onde ficam o checkpoint e o marcador de falha. Precisa ser
        estável entre encarnações do actor — é por ele que a nova encarnação
        encontra o estado da anterior.
    checkpoint_every:
        N. `0` desativa o checkpoint (linha de base sem overhead).
    work_s:
        Duração simulada de uma tool call.
    state_kb:
        Tamanho do payload de estado, em KiB. Controla δ.
    history_len:
        Quantos resultados de ferramenta manter no histórico (limitado, para
        que o tamanho do estado não cresça sem limite ao longo da execução).
    """

    def __init__(
        self,
        workdir: str,
        checkpoint_every: int,
        work_s: float = 0.002,
        state_kb: int = 512,
        history_len: int = 64,
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.workdir / "agent.ckpt"
        self.crash_marker = self.workdir / "crash.json"

        self.checkpoint_every = int(checkpoint_every)
        self.work_s = float(work_s)
        self.state_kb = int(state_kb)
        self.history_len = int(history_len)

        # Métricas desta encarnação (zeram a cada restart, de propósito:
        # o driver é quem agrega o custo ao longo de toda a execução).
        self.t_work = 0.0
        self.t_ckpt = 0.0
        self.n_ckpt = 0
        self.incarnation_pid = os.getpid()

        # --- recuperação: este é o ponto central do padrão recomendado ---
        t0 = time.perf_counter()
        if self.ckpt_path.exists():
            with open(self.ckpt_path, "rb") as f:
                self.state = pickle.load(f)
            self.restored = True
        else:
            self.state = AgentState(
                scratch=os.urandom(self.state_kb * 1024),
                tool_results=[],
            )
            self.restored = False
        self.t_restore = time.perf_counter() - t0
        # Quantas chamadas existiam no estado com que esta encarnação nasceu.
        # O driver compara com quantas ele já tinha visto concluídas para
        # medir o trabalho perdido.
        self.calls_at_birth = self.state.calls_done

    # -- injeção de falha ---------------------------------------------------

    def _crash_log(self) -> list[int]:
        if self.crash_marker.exists():
            try:
                return json.loads(self.crash_marker.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _should_crash(self, call_idx: int, crash_at: int | None, crash_prob: float) -> bool:
        """Decide se esta chamada deve matar o processo.

        O marcador em disco garante que uma chamada **já** não possa derrubar
        o actor duas vezes. Sem isso, o retry automático (`max_task_retries`)
        re-executaria a mesma chamada com os mesmos argumentos e entraria em
        laço infinito de crash.
        """
        already = self._crash_log()
        if call_idx in already:
            return False
        if crash_at is not None and call_idx == crash_at:
            return True
        if crash_prob > 0.0 and random.random() < crash_prob:
            return True
        return False

    def _record_crash(self, call_idx: int) -> None:
        log = self._crash_log()
        log.append(call_idx)
        # Precisa estar durável *antes* do os._exit, senão a encarnação
        # seguinte não sabe que esta chamada já derrubou o actor.
        with open(self.crash_marker, "w") as f:
            json.dump(log, f)
            f.flush()
            os.fsync(f.fileno())

    # -- checkpoint ---------------------------------------------------------

    def _save_checkpoint(self) -> float:
        """Gravação atômica e durável do estado. Devolve o custo em segundos.

        `fsync` + `os.replace` é o que torna δ realista: um checkpoint que
        fica no cache da página não sobrevive à queda do nó, e a doc do Ray
        alerta que o checkpoint precisa estar acessível a todo o cluster
        (aqui, execução de um nó só, o disco local faz esse papel).
        """
        t0 = time.perf_counter()
        tmp = self.ckpt_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.ckpt_path)
        dt = time.perf_counter() - t0
        self.t_ckpt += dt
        self.n_ckpt += 1
        return dt

    # -- API ----------------------------------------------------------------

    def step(
        self,
        call_idx: int,
        crash_at: int | None = None,
        crash_prob: float = 0.0,
    ) -> dict:
        """Executa uma tool call e, se for a hora, faz checkpoint."""
        if self._should_crash(call_idx, crash_at, crash_prob):
            self._record_crash(call_idx)
            # Morte abrupta do processo: o raylet detecta, avisa o GCS, e o
            # GCS ressubmete a creation task deste actor (max_restarts=-1).
            os._exit(1)

        t0 = time.perf_counter()
        busy_work(self.work_s)
        self.state.calls_done += 1
        self.state.tool_results.append({"i": call_idx, "ok": True})
        if len(self.state.tool_results) > self.history_len:
            del self.state.tool_results[: -self.history_len]
        self.t_work += time.perf_counter() - t0

        ckpt_s = 0.0
        if self.checkpoint_every and self.state.calls_done % self.checkpoint_every == 0:
            ckpt_s = self._save_checkpoint()

        return {
            "calls_done": self.state.calls_done,
            "calls_at_birth": self.calls_at_birth,
            "restored": self.restored,
            "checkpointed": ckpt_s > 0.0,
            "pid": self.incarnation_pid,
        }

    def probe(self) -> int:
        """Chamada vazia: mede a latência de RPC do actor (sem trabalho)."""
        return self.incarnation_pid

    def stats(self) -> dict:
        return {
            "calls_done": self.state.calls_done,
            "calls_at_birth": self.calls_at_birth,
            "restored": self.restored,
            "t_work": self.t_work,
            "t_ckpt": self.t_ckpt,
            "n_ckpt": self.n_ckpt,
            "t_restore": self.t_restore,
            "pid": self.incarnation_pid,
            "ckpt_bytes": self.ckpt_path.stat().st_size if self.ckpt_path.exists() else 0,
            "checksum": self.state.checksum(),
        }
