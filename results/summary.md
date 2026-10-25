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
