# Qual é o intervalo ótimo de checkpoint para um actor Ray?

Estudo empírico, com Ray real, do trade-off entre **custo de fazer checkpoint** e
**trabalho perdido quando o actor morre**. O cenário simulado é um agente que
executa chamadas de ferramenta (*tool calls*) em sequência, acumulando estado —
o caso de uso que hoje domina o uso de actors no Ray.

Trabalho da disciplina de Processamento Paralelo e Distribuído.

---

## 1. A lacuna

O paper original do Ray (Moritz et al., *Ray: A Distributed Framework for
Emerging AI Applications*, OSDI 2018) apresenta o modelo de tolerância a falhas
baseado em **linhagem**: se um objeto é perdido, o Ray reexecuta a tarefa que o
produziu. O paper mede throughput e tempo de recuperação de tarefas, mas não
trata do estado de actors com a mesma profundidade — e a arquitetura descrita
lá já está bastante distante do Ray 2.x.

O whitepaper atual, **Ray v2 Architecture**, é explícito sobre o limite dessa
abordagem. Em *Object Failure → Large objects and lineage reconstruction*:

> *"Tasks are assumed to be deterministic and idempotent. Thus, by default,
> **objects created by actor tasks are not reconstructable**."*

E em *Actor death*:

> *"Ray also provides an option (`max_restarts`) to automatically restart actors,
> up to a specified number of times. [...] the GCS service will attempt to
> restart a crashed actor by resubmitting its creation task."*

A documentação de *Actor Fault Tolerance* fecha o raciocínio: ao reiniciar, o
estado do actor **"will be recreated by rerunning its constructor"**. Ou seja:

| | objetos de tarefas | estado de actor |
|---|---|---|
| recuperação automática | sim (reconstrução por linhagem) | **não** |
| mecanismo | reexecução da tarefa | checkpoint manual da aplicação |
| frequência recomendada | n/a | **não especificada** |

O Ray reinicia o processo do actor e reexecuta a chamada que falhou, mas **o
estado volta ao que estava no último checkpoint que a aplicação gravou**. Nem o
paper de 2018, nem o whitepaper v2, nem a documentação dizem *com que
frequência* esse checkpoint deve ser feito. É exatamente a pergunta que a
literatura clássica de HPC resolveu para checkpoint de aplicações
(Young 1974, Daly 2006) e que ninguém transpôs para o modelo de actors.

**Este trabalho mede esse trade-off no Ray real e testa se as fórmulas de
Young e Daly continuam valendo nesse contexto.**

## 2. As três perguntas

1. **Overhead de checkpoint vs. intervalo** — quanto cada frequência de
   checkpoint deixa a execução mais lenta? (→ `fig1_overhead.png`)
2. **Trabalho perdido vs. intervalo** — quantas chamadas se perdem, em média,
   se o actor cair antes do próximo checkpoint? (→ `fig2_lostwork.png`)
3. **Custo total esperado vs. intervalo** — somando os dois efeitos, existe um
   ponto de equilíbrio? (→ `fig3_totalcost.png`, validado em
   `fig4_validacao.png`)

## 3. O que é medido, e como

### O actor (`rayckpt/agent.py`)

`ToolCallingAgent` implementa o padrão **self-managed checkpointing** que a
documentação atual recomenda:

```python
@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
class ToolCallingAgent:
    def __init__(self, workdir, checkpoint_every, work_s, state_kb):
        ...
        if self.ckpt_path.exists():          # <- a recuperação acontece aqui,
            self.state = pickle.load(...)    #    porque é o construtor que o
            self.restored = True             #    GCS reexecuta no restart
```

- **uma tool call por chamada**: `busy_work(work_s)` queima CPU de verdade
  (não `sleep`), para que o actor concorra por um núcleo como um rollout real.
- **checkpoint a cada N chamadas**: `pickle` → `fsync` → `os.replace` atômico.
  O `fsync` é o que torna δ realista — um checkpoint que fica no cache de
  página não sobrevive à queda do nó.
- **estado de tamanho controlado** (`state_kb`): um payload fixo representando
  a memória de trabalho do agente. É o parâmetro que varia δ sem mudar nada
  mais, e δ é justamente o que desloca o ótimo nas fórmulas.

### A falha

Não é simulada. O método chama `os._exit(1)`, matando o processo do worker.
O raylet detecta a morte, publica o evento no GCS, e o GCS ressubmete a
*creation task* do actor (`max_restarts=-1`). Um marcador em disco (com `fsync`
antes do `_exit`) garante que a mesma chamada não derrube o actor duas vezes —
sem ele, o retry automático reexecutaria a chamada com os mesmos argumentos e
entraria em laço infinito de crash.

O trabalho perdido é **medido**, não calculado: o driver sabe quantas chamadas
tinha visto concluídas, e a nova encarnação informa com quantas nasceu
(`calls_at_birth`, lido do checkpoint). A diferença é o que o rollback apagou.

### Os experimentos

| Experimento | Arquivo | O que faz |
|---|---|---|
| **A** overhead | `exp_overhead.py` | varre N ∈ {1,2,5,10,20,50,100,200, sem checkpoint} × estado ∈ {64, 512, 4096} KiB, sem falhas. A linha de base sem checkpoint dá `w`; a diferença dá o overhead |
| **B** trabalho perdido | `exp_failure.py` | para cada N, ensaios independentes com **uma** falha real em ponto uniforme do ciclo; mede chamadas perdidas e `R` |
| **C** custo total | `analysis.py` | combina δ, w e R medidos no modelo de custo e localiza o mínimo; compara com Young e Daly |
| **D** validação | `exp_endtoend.py` | falhas como **processo de Poisson** ao longo de toda a execução, com o trabalho perdido **refeito**; mede o tempo de parede real por chamada útil e confere se o mínimo cai onde o modelo previu |

Diferença importante entre B e D: em B a falha é única e o trabalho perdido
*não* é refeito (o objetivo é isolar quanto se perde). Em D o driver volta ao
ponto do checkpoint e reexecuta, que é o que Young e Daly contam como *rework*
no tempo de parede. O laço de D é guiado pelo contador que o actor devolve, não
por um contador do driver — assim o rewind acontece naturalmente.

### MTBF escalado

Um nó real falha em horas ou dias; esperar por isso tornaria o experimento
inviável. Em D o MTBF é escalado para a casa dos segundos. O que importa para
testar as fórmulas é a razão adimensional ξ = √(δ/2M), que é preservada quando
δ e M são escalados juntos — é essa razão que governa as equações de Daly.

## 4. O modelo analítico (`rayckpt/model.py`)

Notação de Daly (2006): τ intervalo de computação entre checkpoints, δ custo de
um checkpoint, M o MTBF, R o custo de reinício.

O custo de primeira ordem decompõe exatamente nas três perguntas do trabalho:

```
T_w = T_s · [ 1 + δ/τ + (τ/2 + R)/M ]
             ───   ───   ───────────
          trabalho  overhead   trabalho perdido
             útil      (↓N)     + reinício (↑N)
```

Minimizando em τ:

| | fórmula | fonte |
|---|---|---|
| Young (1974) | τ* = √(2δM) | CACM 17(9), p. 530 |
| Daly 1ª ordem | τ* = √(2δ(M+R)) | FGCS 22(3), eq. (12) |
| Daly alta ordem | τ* = √(2δM)·[1 + ⅓√(δ/2M) + ⅑(δ/2M)] − δ, para δ < 2M;  M caso contrário | eq. (37) |
| Daly exato | τ* = M·η − δ, com η = 2ξ²+1+W(−e^(−2ξ²−1)), ξ = √(δ/2M) | eq. (25) |

Em unidades de chamada, N* = τ*/w.

Dois detalhes da literatura que o experimento permite checar:

- Daly conclui que, apesar de a primeira ordem colocar R sob o radical, **"the
  higher order model demonstrates that in fact R has no contribution"** ao
  ótimo. Como medimos R diretamente (e ele é grande: ~1,5 s, dominado pelo
  restart do processo pelo GCS), dá para ver isso acontecendo.
- A solução de alta ordem (eq. 37) garante erro relativo em T_w abaixo de
  **0,2%**; a de ordem mais baixa, equivalente a Young, fica abaixo de **5%**
  mesmo no pior caso δ = M/2.

## 5. Como rodar

Requer [`uv`](https://docs.astral.sh/uv/). Ray 2.59 em Python 3.14.

```bash
uv sync

# tudo (~25 min): experimentos A, B, D e depois a análise
uv run python -m rayckpt.run_all

# ou cada parte separadamente
uv run python -m rayckpt.exp_overhead     # A
uv run python -m rayckpt.exp_failure      # B
uv run python -m rayckpt.exp_endtoend     # D
uv run python -m rayckpt.analysis         # figuras + results/summary.md

# versão rápida de cada um, para verificar que está tudo de pé (~1 min)
uv run python -m rayckpt.run_all --quick
```

Saídas:

- `results/*.csv` — dados brutos, uma linha por execução/ensaio
- `results/summary.md` — parâmetros medidos e tabelas comparativas
- `figures/*.png` — as quatro figuras

## 6. Resultados

Números completos em `results/summary.md` (gerado); apresentação em
`slides/apresentacao.html`. Resumo do que saiu nesta máquina
(macOS, 10 núcleos, Ray 2.59, Python 3.14):

### Parâmetros medidos

| | valor |
|---|---|
| `w` — uma tool call (trabalho + RPC do Ray) | 2,45 ms |
| `δ` — um checkpoint, estado de 64 / 512 / 4096 KiB | 0,71 / 0,88 / 2,59 ms |
| `R` — recuperação completa (mediana de 180 falhas) | 1481 ms |
| — dentro de R, a leitura do checkpoint | 0,28 ms |

**O custo de recuperação no Ray é quase todo restart de processo, não I/O de
estado**: a leitura do checkpoint é ~5200× menor que o restart do actor pelo
GCS. Isso inverte a intuição de HPC, onde o restart é dominado por reler o
dump, e tem uma consequência prática — engordar o estado do agente encarece o
*checkpoint* muito mais do que encarece a *recuperação*.

### As três perguntas

1. **Overhead** cai como δ/N, como previsto, até encostar no piso de resolução
   da medida (±0,011 ms). Com estado de 4 MiB, checkpoint a cada chamada custa
   **136%** de overhead sobre o trabalho útil; a cada 5, 17%; a cada 50, 3,0%;
   a cada 200, 0,8%. Com estado de 64 KiB os mesmos pontos são 17% / 2,5% /
   1,3% / ~0%.
2. **Trabalho perdido** segue (N−1)/2 em toda a faixa testada: nenhum N se
   afasta da teoria por mais de 1,7 erro padrão (180 falhas reais, 30 por N).
3. **Custo total** tem mínimo. Com os parâmetros acima:

| MTBF | N\* Young | N\* Daly alta ordem | custo no ótimo |
|---|---|---|---|
| 2 s | 41 | 41 | +79% |
| 1 min | 227 | 227 | +3,4% |
| 1 h | 1760 | 1759 | +0,16% |

### Validação ponta a ponta

194 falhas reais injetadas como processo de Poisson, 8381 chamadas refeitas
após rollback, 18 execuções de 1200 chamadas por ponto. Ajustando
custo(N) = a + b/N + c·N aos dados (N\* = √(b/c)):

| curva | N\* ajustado | IC 95% | previsto |
|---|---|---|---|
| tempo de parede bruto | 36 | [24, 123] | 44 |
| descontado o restart | 32 | [28, 44] | 44 |

O previsto cai dentro do IC 95% nas duas curvas, e os coeficientes do ajuste
recuperam os parâmetros medidos dentro de um fator ~1,5 (`b` = 2,02 ms contra
δ = 2,59 ms; `c` = 0,0020 contra w²/2M = 0,0013). **O modelo de Young/Daly, de
1974, descreve um actor Ray sem nenhuma adaptação.**

### Duas ressalvas metodológicas

- **O tempo de parede bruto é ruim para localizar o ótimo.** O termo de
  restart (1,48 s por falha, contagem de falhas seguindo Poisson) é
  independente de N e carrega quase toda a variância: o `argmin` da grade cai
  em N=200, que é ruído. Descontar `n_falhas × R` — ambos medidos — é legítimo
  exatamente porque Daly mostra que R não desloca o ótimo, e é o que deixa o
  trade-off visível. As duas curvas estão no painel da `fig4`.
- **A figura compara com o modelo de primeira ordem, não com a forma exata de
  Daly.** A forma exata traz um fator e^(R/M) que pressupõe falhas atingindo a
  aplicação também durante o downtime. Aqui a falha só é sorteada dentro de
  `step()`, isto é, só durante computação — que é a hipótese sob a qual vale a
  expressão de primeira ordem. Com R = 1,48 s e M = 2 s esse fator vale
  e^0,74 ≈ 2,1, suficiente para deslocar o *nível* da curva em mais de 20% sem
  mudar onde está o mínimo.

## 7. Limitações

### Um nó, sem containers nem Kubernetes

O cluster é local: `ray.init(num_cpus=4)` sobe o GCS, o raylet e os workers
como processos desta máquina. Não há Docker, KubeRay ou rede entre nós. O que
isso cobre e o que não cobre:

| | medido aqui | cluster real |
|---|---|---|
| actor como processo de SO separado | sim | sim |
| morte do **worker** e restart pelo GCS | sim, real (`os._exit`) | sim |
| morte do **nó** | não | o caso que mais dói |
| rede entre nós | não (loopback) | latência, partição |
| checkpoint em storage compartilhado | não (SSD local) | exigido pela doc do Ray |
| startup do worker | processo local | contêiner + runtime env |

O caminho de falha em si é real e completo — `os._exit(1)` mata o processo, o
raylet detecta, publica no GCS, o GCS ressubmete a creation task, o construtor
relê o checkpoint — e é esse caminho inteiro que os ~1,5 s de R cronometram.
O que falta é a dimensão de **nó**.

**Consequência prevista.** Num cluster real δ cresce bastante: um checkpoint de
4 MiB em S3 ou NFS custa dezenas de milissegundos, não os ~2,6 ms do SSD local.
Como N\* = √(2δM)/w e Daly mostra que R *não* entra no ótimo, a previsão é que
**o intervalo ótimo cresça com √δ**: passar de δ = 2,6 ms para δ = 80 ms
multiplica N\* por √(80/2,6) ≈ 5,5. Com MTBF de 1 h, o N\* ≈ 1760 medido aqui
viraria ≈ 9700. A varredura de três tamanhos de estado (64 / 512 / 4096 KiB) do
Experimento A é a evidência parcial disso: δ escala com o estado, e o ótimo
acompanha.
- **Estado de tamanho fixo.** δ é mantido ~constante de propósito, porque é a
  hipótese dos modelos de Young e Daly. Um agente real acumula contexto, então
  δ cresce ao longo da execução — o ótimo passaria a ser uma função do tempo, e
  é aí que entram os trabalhos de checkpoint adaptativo citados abaixo.
- **Falhas independentes e de Poisson.** É a hipótese de Young e Daly. Falhas
  correlacionadas (perda de nó levando vários actors) não são cobertas.
- **MTBF escalado**, pelos motivos da seção 3.
- **R é dominado pelo restart do processo**, que no Ray é da ordem de 1,5 s
  numa máquina local. Em cluster, com runtime environments, tende a ser maior.

## 8. Referências

- Moritz, P. et al. (2018). *Ray: A Distributed Framework for Emerging AI
  Applications.* OSDI '18. https://www.usenix.org/conference/osdi18/presentation/moritz
- *Ray v2 Architecture* (whitepaper atual). Seções *Failure Model*,
  *Lifetime of an Actor*, *Actor death*, *Object Failure*, *Global Control
  Service → Fault tolerance*.
- Ray docs — *Actor Fault Tolerance*.
  https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html
- Young, J. W. (1974). *A first order approximation to the optimum checkpoint
  interval.* Communications of the ACM 17(9):530–531.
  https://dl.acm.org/doi/10.1145/361147.361115
- Daly, J. T. (2006). *A higher order estimate of the optimum checkpoint
  interval for restart dumps.* Future Generation Computer Systems 22(3):303–312.
  https://doi.org/10.1016/j.future.2004.11.016
- Benoit, A. et al. (2022). *Checkpointing à la Young/Daly: An Overview.*
  https://dl.acm.org/doi/10.1145/3549206.3549328
- Benoit, A. et al. (2024). *A survey on checkpointing strategies: Should we
  always checkpoint à la Young/Daly?* Future Generation Computer Systems.
  https://doi.org/10.1016/j.future.2024.107701
- Strati, F. et al. (2025). *PCcheck: Persistent Concurrent Checkpointing for
  ML.* ASPLOS '25. https://anakli.inf.ethz.ch/papers/PCcheck_asplos25.pdf
