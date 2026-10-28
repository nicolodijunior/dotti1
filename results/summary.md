# Resumo dos resultados

## Parâmetros medidos

| Parâmetro | Valor | Como foi medido |
|---|---|---|
| w (custo de uma chamada) | 2.452 ms | tempo de parede por chamada com checkpoint desligado (N=0) |
| δ (checkpoint, estado 64 KiB) | 0.709 ms | tempo de pickle + fsync + rename atômico, dentro do actor |
| δ (checkpoint, estado 512 KiB) | 0.883 ms | tempo de pickle + fsync + rename atômico, dentro do actor |
| δ (checkpoint, estado 4096 KiB) | 2.587 ms | tempo de pickle + fsync + rename atômico, dentro do actor |
| R (recuperação) | 1481 ms (IQR 1437–1535) | latência da chamada que morreu, menos a latência de uma chamada normal |
| — leitura do checkpoint | 0.3 ms | tempo dentro do construtor da nova encarnação |

O custo de recuperação é dominado pelo restart do processo do actor pelo GCS (1481 ms), não pela leitura do estado (0.28 ms): a leitura é 5209x menor que o restart. Isso tem uma consequência prática — aumentar o estado do agente encarece o checkpoint (δ) muito mais do que encarece a recuperação.

## Trabalho perdido vs. intervalo

| N | perdidas (média medida) | (N−1)/2 | erro padrão | desvio em σ | ensaios |
|---|---|---|---|---|---|
| 1 | 0.00 | 0.0 | 0.00 | +0.0 | 30 |
| 5 | 1.87 | 2.0 | 0.27 | -0.5 | 30 |
| 10 | 4.83 | 4.5 | 0.51 | +0.7 | 30 |
| 20 | 10.27 | 9.5 | 0.92 | +0.8 | 30 |
| 50 | 25.53 | 24.5 | 2.89 | +0.4 | 30 |
| 100 | 39.67 | 49.5 | 5.77 | -1.7 | 30 |

Nenhum N se afasta da previsão (N−1)/2 por mais de 1.7 erros padrão, ou seja, o trabalho perdido medido é compatível com a teoria em toda a faixa. A dispersão é intrinsecamente alta — o ponto de falha é uniforme no ciclo, então o desvio-padrão da perda cresce com N (≈ N/√12), e é por isso que N grande precisa de mais ensaios para a média estabilizar.

## Intervalo ótimo previsto (δ do maior estado)

δ = 2.59 ms, w = 2.45 ms, R = 1481 ms

| MTBF | N* Young (1974) | N* Daly 1ª ordem | N* Daly alta ordem | N* Daly exato | mínimo numérico | custo no ótimo |
|---|---|---|---|---|---|---|
| 2 s | 41 | 55 | 41 | 41 | 42 | +79.14% |
| 60 s | 227 | 230 | 227 | 227 | 227 | +3.40% |
| 3600 s | 1760 | 1760 | 1759 | 1759 | 1764 | +0.16% |

Young e Daly 1ª ordem diferem porque só o segundo coloca R sob o radical. Daly (2006) mostra na solução de alta ordem que R **não** contribui para o ótimo — e de fato a coluna de alta ordem fica junto do mínimo numérico, enquanto a de 1ª ordem se afasta quando R é grande em relação ao MTBF.

## Validação ponta a ponta

MTBF escalado para 2 s; δ = 2.59 ms, w = 2.30 ms, R = 1481 ms.

- falhas reais injetadas e recuperadas: **194**
- chamadas refeitas após rollback: **8381**
- previsto por Young: N* = 44; Daly alta ordem: N* = 44; Daly exato: N* = 44


| Curva | N* do ajuste | IC 95% | argmin da grade |
|---|---|---|---|
| tempo de parede bruto | 36 | [24, 123] | 200 |
| descontado o restart | 32 | [28, 44] | 25 |

O ajuste é da forma custo(N) = a + b/N + c·N, que é o modelo de primeira ordem em unidades de chamada; N* = √(b/c). Se o modelo descreve os dados, `b` recupera δ e `c` recupera w²/(2·MTBF):

| coeficiente | ajustado (sem restart) | esperado | significado |
|---|---|---|---|
| a | 2.398 ms | 2.299 ms | trabalho útil w |
| b | 2.019 ms | 2.587 ms | custo de um checkpoint δ |
| c | 0.00197 ms | 0.00132 ms | trabalho perdido w²/(2·MTBF) |

O tempo de parede **bruto** é dominado pelo termo de restart (1481 ms por falha, com a contagem de falhas seguindo Poisson), que é independente de N e carrega quase toda a variância — por isso seu argmin na grade não é confiável. Descontar `n_falhas × R` (ambos medidos) é legítimo justamente porque Daly mostra que R não desloca o ótimo, e deixa visível o trade-off que interessa.

| N | bruto (ms/chamada útil) | erro padrão | sem restart | erro padrão | execuções | falhas |
|---|---|---|---|---|---|---|
| 1 | 6.461 | 0.378 | 4.747 | 0.114 | 18 | 25 |
| 5 | 4.538 | 0.359 | 2.755 | 0.020 | 18 | 26 |
| 10 | 4.262 | 0.367 | 2.617 | 0.017 | 18 | 24 |
| 25 | 4.208 | 0.432 | 2.563 | 0.021 | 18 | 24 |
| 50 | 4.100 | 0.289 | 2.592 | 0.035 | 18 | 22 |
| 100 | 4.734 | 0.400 | 2.815 | 0.116 | 18 | 28 |
| 200 | 3.926 | 0.414 | 2.760 | 0.122 | 18 | 17 |
| 400 | 4.962 | 0.456 | 3.043 | 0.117 | 18 | 28 |

## Projeção para storage compartilhado

Todo o δ medido acima é de **disco local**, que é o piso otimista: a própria documentação do Ray pede que o checkpoint esteja acessível a todo o cluster, e nesse caso δ é dominado por latência e vazão de rede. Ajustando o δ medido nos três tamanhos de estado ao modelo δ = latência + bytes/vazão, o nosso storage é **0.66 ms + 2181 MB/s** — compatível com NVMe local, e uma a duas ordens de grandeza mais rápido que qualquer storage compartilhado.

A tabela abaixo é **projeção, não medição**: aplica Young/Daly (já validados contra o ótimo medido na seção anterior) ao δ de cada cenário, com MTBF de 1 h e w = 2.45 ms. Isso é legítimo porque a validação é adimensional — depende de ξ = √(δ/2M), não dos valores absolutos.

| estado | storage | δ projetado | δ/δ_local | τ* (Daly exato) | N* se w = 2 s | custo no ótimo |
|---|---|---|---|---|---|---|
| 1 MiB | EFS / NFS | 15.3 ms | 13x | 0.2 min | 5 | +0.33% |
| 1 MiB | S3 (stream único) | 39.9 ms | 35x | 0.3 min | 8 | +0.51% |
| 1 MiB | S3 (cross-region) | 138.6 ms | 121x | 0.5 min | 16 | +0.92% |
| 8 MiB | EFS / NFS | 48.1 ms | 11x | 0.3 min | 9 | +0.56% |
| 8 MiB | S3 (stream único) | 104.4 ms | 23x | 0.5 min | 14 | +0.80% |
| 8 MiB | S3 (cross-region) | 264.3 ms | 59x | 0.7 min | 22 | +1.25% |
| 64 MiB | EFS / NFS | 309.9 ms | 10x | 0.8 min | 24 | +1.35% |
| 64 MiB | S3 (stream único) | 620.7 ms | 20x | 1.1 min | 33 | +1.90% |
| 64 MiB | S3 (cross-region) | 1269.9 ms | 40x | 1.6 min | 47 | +2.70% |
| 512 MiB | EFS / NFS | 2404.3 ms | 10x | 2.2 min | 65 | +3.70% |
| 512 MiB | S3 (stream único) | 4750.8 ms | 19x | 3.0 min | 91 | +5.18% |
| 512 MiB | S3 (cross-region) | 9314.7 ms | 38x | 4.2 min | 126 | +7.24% |

A coluna em tempo (τ*) é a que não depende do agente; N* em chamadas só é comparável a w fixo, e o w = 2.45 ms das nossas chamadas sintéticas é duas a três ordens de grandeza menor que uma tool call real (uma chamada de LLM ou de API leva segundos). A coluna `N* se w = 2 s` traduz τ* para um agente realista, e é onde o resultado fica acionável: algumas dezenas de chamadas, não centenas.

Duas consequências. Primeira, o ótimo se desloca com √δ: sair do disco local para S3 multiplica δ por uma ordem de grandeza e portanto N* por ~3 — checkpointar na frequência que o disco local sugere é checkpointar demais. Segunda, cada storage tem um ponto de cruzamento em latência×vazão (EFS / NFS ≈ 2.4 MiB, S3 (stream único) ≈ 3.4 MiB, S3 (cross-region) ≈ 6.9 MiB) abaixo do qual δ é ~constante e não depende do tamanho do estado; os três tamanhos que medimos caem todos aí, e é por isso que a Figura 1 mostra δ quase insensível ao estado nos dois tamanhos menores.

### Validação da projeção com δ remoto injetado

A projeção acima aplica as fórmulas fora do regime em que elas foram validadas, então ela foi testada: o Experimento C/D foi repetido com o custo de storage compartilhado injetado de verdade no caminho crítico do actor (30 ms de latência + 120 MB/s, perfil S3), o que leva δ de 2.59 ms para **67.5 ms — 26x** o disco local.

- ótimo previsto por Daly com esse δ: **N* = 206**
- ótimo medido (ajuste): **N* = 184**, IC 95% [152, 302]
- falhas reais injetadas: 114; chamadas refeitas: 13100


| coeficiente | ajustado | esperado | significado |
|---|---|---|---|
| a | 2.620 ms | 2.313 ms | trabalho útil w |
| b | 70.9 ms | 67.5 ms | custo de um checkpoint δ |
| c | 0.00210 ms | 0.00134 ms | trabalho perdido w²/(2·MTBF) |

O previsto cai dentro do IC 95% do medido (0.89x o previsto), e o coeficiente `b` do ajuste recupera δ a partir dos dados (70.9 ms contra 67.5 ms previstos pelo modelo de dois termos). Ou seja, a projeção da tabela acima não é extrapolação cega: o modelo continua descrevendo o custo quando δ é dominado pela latência do storage, e não pelo disco.

Ver `figures/fig5_validacao_remota.png`.

| N | custo sem restart (ms/chamada útil) | erro padrão | execuções | falhas |
|---|---|---|---|---|
| 25 | 5.500 | 0.038 | 10 | 12 |
| 50 | 4.182 | 0.050 | 10 | 19 |
| 100 | 3.510 | 0.049 | 10 | 22 |
| 200 | 3.371 | 0.151 | 10 | 17 |
| 300 | 3.558 | 0.232 | 10 | 22 |
| 600 | 4.096 | 0.443 | 10 | 22 |
