# SITUAÇÃO ATUAL DO PROJETO — BBTAN Bot

Bot em Python que joga o BBTAN (brick-breaker mobile) sozinho, num emulador Android
no Windows. Usa visão computacional (OpenCV + mss) para ler o estado do jogo e
PyAutoGUI para executar os disparos.

- **Branch refletida:** `dev` — último commit `53ff021` (merge do PR #17)
- **`main`:** `b80e087` (merge do PR #9 — README com diagrama e GIF)
- **Data desta atualização:** 2026-09-29

> A versão anterior deste documento refletia o commit `df96afa` (2026-07-26) e
> estava defasada em quase tudo (detecção de rodada por regiões rosa, detecção de
> jogo travado, `ROW_MULTIPLIERS` antigos, coordenadas absolutas). Esta revisão foi
> escrita a partir do código do `dev`.

---

## Modelo de branches

- `main`: versão estável (o que aparece no GitHub/CV).
- `dev`: integração. Toda branch de trabalho sai do `dev` e volta por PR
  (`gh pr create --base dev`). O `main` só recebe o `dev` quando decidido.
- Fluxo por correção: branch → implementa + valida offline → push → PR → merge
  (sem conflito e com validação ok) → apaga a branch.

---

## Resumo — status dos módulos

| Módulo | Papel | Status |
|--------|-------|--------|
| `config.py` | Constantes, regiões, HSV, pesos | ✅ Toda geometria escala a partir de `GAME_REGION` (`ESCALA`, `px()`, `area_px()`); HUD relativo ao tabuleiro. |
| `vision.py` | Captura e detecção | ✅ Contador de bolas com filtro de coerência entre turnos; HP ilegível tratado como desconhecido; power-up roxo detectado; espera o tabuleiro parar. |
| `physics.py` | Simulação e escolha de ângulo | ✅ Funcional. Devolve termos do score e HP final previsto por bloco. ⚠️ Escolhe tiros frágeis (ver análise do `bot_1`). |
| `executor.py` | Disparo (swipe) | ✅ Funcional. ⚠️ Tiro sai ~1,3° mais inclinado que o escolhido. |
| `main.py` | Loop principal | ✅ Funcional. ⚠️ Coordenadas de restart ainda absolutas; `--debug`/`--smart-wait` sem efeito. |
| `log_turno.py` | Diagnóstico por turno | ✅ Log + imagens por turno em `logs/bot_N/` (previsto × real × conferência). |
| `imagem.py` | Conferência da calibração | ✅ Print com todas as posições do `config.py` desenhadas. |
| `desenhar_caminho.py` | Caminho ideal na tela atual | ✅ Salva em `prints/`. |
| `main_vision.py` / `monitor.py` | Ferramentas de visão | ✅ |
| `tools/*` | Calibração, templates, debug | ✅ (⚠️ `snapshot.py`/`debug_view.py` só rodam com `python -m tools.x`) |
| Testes | — | ❌ Inexistentes. A validação é feita com replays de `logs/` e testes sintéticos pontuais. |

---

## Fluxo de um turno

1. `vision.esperar_tabuleiro_parado()` — compara a grade a cada 0,2 s até parar de
   mudar (teto 4 s). Substituiu os `sleep(3)` + `sleep(2.0)` fixos.
2. `scan_game()` — captura o `GAME_REGION`, detecta blocos, power-ups e bola, lê o
   contador de bolas e testa game over (tela >25% clara). HP ilegível já sai
   estimado (`ajustar_hp` sem referência).
3. No `main`: filtro de coerência do contador (`escolher_contagem`), ajuste de HP
   com o turno anterior (`ajustar_hp`) e conferência da previsão anterior.
4. `run_turn` → `find_best_angle` simula os ângulos 19..161 (passo 1°) e re-simula o
   melhor com `record_path=True`; o log mostra a decisão e salva `turno_NNN_previsto.png`.
5. `executor.execute_shot` — swipe a partir da bola + clique no acelerar.
6. `log_turno.capturar_rastro` — grava o voo real por até 2,5 s, vigiando o botão
   retrieve; para quando a rodada acaba. Salva `turno_NNN_real.png` e mede o ângulo real.
7. `aguardar_fim_da_rodada` — só espera se o retrieve ainda estiver na tela; após
   `TEMPO_MAX_RODADA` (180 s) clica nele para forçar o fim.

Game over: grava em `historico.txt`, clica 3× na sequência de restart e zera o estado.

---

## Estrutura de arquivos

```
bbtan_bot/
├── config.py            — Constantes, escala, regiões, HSV, pesos
├── vision.py            — Captura + detecção (blocos / HP / power-ups / bola / contador)
├── physics.py           — Simulação de trajetória + escolha de ângulo
├── executor.py          — Disparo via PyAutoGUI
├── main.py              — Loop principal
├── log_turno.py         — Diagnóstico por turno (log + imagens em logs/bot_N/)
├── imagem.py            — Print com as posições do config desenhadas (calibração)
├── desenhar_caminho.py  — Caminho ideal na tela atual (prints/)
├── main_vision.py       — Loop só de visão
├── monitor.py           — Monitor ao vivo
├── historico.txt        — Turnos sobrevividos por partida (gitignored)
├── docs/                — demo.gif, calibracao.png, este documento, notas
├── templates/digits/    — Templates 28px do contador de bolas
├── templates/hp_digits/ — Templates 20px do HP (+ amostras brutas)
├── tools/               — Calibração, captura de templates, janelas de debug
├── logs/                — Sessões do main.py (gitignored)
└── prints/              — Saída do desenhar_caminho.py / imagem.py (gitignored)
```

---

## config.py

### Escala e posição

Todos os valores em pixel foram medidos com o jogo em `REF_LARGURA = 452` px.
`ESCALA = GAME_REGION[2] / 452`; comprimentos passam por `px()`, áreas por
`area_px()`. O HUD (contador, retrieve, acelerar, clique do retrieve) é relativo ao
canto do `GAME_REGION` (`regiao_hud`/`ponto_hud`). Mudar a janela de lugar = corrigir
só o `x, y` do `GAME_REGION` (conferir com `python imagem.py`).

```python
GAME_REGION = (1352, 244, 452, 578)
COL_LEFTS/COL_RIGHTS, _GRID_TOP_Y = 67, LEVEL_BOTS[0] = 513, WALL_* — via px()
BALL_RADIUS = px(9)
SIM_PATH_BUDGET = 4000 * ESCALA; SIM_STEP_SIZE = 1.0 * ESCALA  # custo não muda com o tamanho
DRAG_DISTANCE = px(150)
```

`row_index 0` = fileira mais baixa (L7, a do game over); `row_index 6` = topo (L1).

### Power-ups (HSV)

| Tipo | Cor | H |
|---|---|---|
| `EXTRA_BALL` (+) | ciano | 81–91 |
| `LASER_H/V` (<>) | azul | 97–107 |
| `BONUS_B` (B) | ouro | 20–26 |
| `RANDOM` (3 setas) | roxo | 131–145 |

Raio aceito: `MIN/MAX_POWERUP_RADIUS = px(7)/px(15)`.

### Scoring

```python
ROW_MULTIPLIERS = {0: 100, 1: 50, 2: 25, 3: 15, 4: 10, 5: 8, 6: 5}
W_HIT = 1.0; W_DESTROY = 5.0; W_EXTRA_BALL = 30.0; W_BONUS_B = 5.0
W_ROW0_SURVIVE = 0.0; W_LOOP = 0.0   # desligados
ANGLE_MIN, ANGLE_MAX = 19, 161        # ANGLE_STEP é decorativo (passo 1 fixo)
TEMPO_MAX_RODADA = 180
```

`score = physics.compute_score(ScoreTerms)`; ranking por `(score, total_hits)`.

---

## vision.py

- **Blocos:** HSV amplo + contornos; big-area split em 3 fases; gate 2x2
  (`_qualifies_as_big_block`). `BLOCK_COLOR_RANGES` tem 11 cores, sem duplicatas.
- **HP (`_read_hp`):** template matching em `templates/hp_digits/`. Falha de leitura
  devolve `HP_DESCONHECIDO` (0) — antes virava 1, e "0" fazia a simulação atravessar
  o bloco. `ajustar_hp(blocks, anterior)`: HP que "subiu" volta ao valor do turno
  anterior; desconhecido vira o HP previsto do bloco ou o maior HP lido. Marca
  `Block.hp_estimado` (o log mostra `?`).
- **Contador de bolas:** `_rank_digit` devolve candidatos por dígito;
  `read_ball_count_template` preenche `CONTADOR_CANDIDATOS`. `escolher_contagem`
  aceita só leituras em `[anterior, anterior + "+" na tela]`; senão tenta trocar um
  dígito ou usa o esperado. Leitura fora da faixa só é aceita se repetir, tiver o
  mesmo número de dígitos e não for troca conhecida (3/8, 5/6, 1/0...).
- **Power-ups:** HSV + circularidade > 0,6 + raio; laser H/V pela projeção.
- **Retrieve:** `detect_retrieve_button` (roxo, ≥30% da região).
- **`esperar_tabuleiro_parado`:** estabilidade da grade (sem o robô).
- `detect_round_end` / `ROUND_END_REGIONS` continuam no código, mas não são usados.

---

## physics.py

- `_sim_one_ball`: AABB para quadrados, reflexão pela normal diagonal nos triângulos.
  Bola que toca o power-up `RANDOM` termina ali (o voo depois é imprevisível); ele
  não é coletado, então afeta todas as bolas.
- `simulate_trajectory`: bolas em sequência com batch (bolas que refazem o mesmo voo
  inteiro); laser e `record_path` forçam batch 1.
- `SimResult` ganhou `terms` (termos crus do score) e `hp_final`
  (`row, col, hp_antes, hp_depois` por bloco) — base da conferência do log.
- `find_best_angle`: 143 simulações + re-simulação do melhor com path.

---

## log_turno.py — diagnóstico

Cada sessão do `main.py` cria `logs/bot_N/` (N = maior + 1) com `sessao.log` e,
por turno, `turno_NNN_previsto.png`, `turno_NNN_real.png` e `turno_NNN_contador.png`.
O texto de cada turno tem: a grade 7×7 lida, a decisão (composição do score, top 5,
vantagem sobre o 2º), o dano previsto por bloco, o rastro real (ângulo real de saída,
% do voo sobre a 1ª bola, primeira divergência) e, no turno seguinte, a conferência
previsto × lido com uma pista do motivo de cada diferença (OCR, visão ou física).

---

## Análise da sessão `logs/bot_1` (52 turnos, antes dos PRs #14–#17)

Baseline para medir as correções:

| Métrica | Valor |
|---|---|
| Dano real / dano previsto | **0,62** |
| Blocos previstos como atingidos que não levaram dano | 72 |
| Blocos com o HP previsto exato | 15 em 52 turnos |
| Leituras erradas do contador | ~12 de 53 (3→8, 5→6, dígitos perdidos) |
| HP lido como 0 | 3 |
| Erro do ângulo real de saída | +1,77° (direita), −0,70° (esquerda), desvio 2,3° |
| Tempo por turno | 16–20 s (~10 s parados) |

Causas identificadas: leitura (contador, HP), execução (viés do ângulo), decisão
(tiros rasos que passam em frestas e só funcionam no ângulo exato) e espera fixa.

---

## Histórico recente (PRs)

| PR | Mudança |
|---|---|
| #10 | Geometria escalável (`ESCALA`, HUD relativo) + `imagem.py` |
| #11 | Painel de trajetória desenha sobre o frame do turno simulado |
| #12 | Diagnóstico por turno (`log_turno.py`) |
| #13 | Power-up roxo (`RANDOM`) + pastas `logs/bot_N` |
| #14 | Ângulo real de saída no log; rastro comparado só com a 1ª bola |
| #15 | Contador de bolas com coerência entre turnos |
| #16 | HP ilegível como desconhecido; HP não sobe |
| #17 | Fim da espera fixa (~10 s por turno) |

---

## Próximos passos

Plano em andamento (correções da análise do `bot_1`):

1. **Checkpoint:** jogar uma sessão nova e comparar com a tabela acima.
2. **`fix/vies-angulo`:** medir o viés com a sessão nova e compensar no executor.
3. **`feat/pontuacao-robusta`:** nota de cada ângulo como média ponderada dos
   vizinhos (±1–2°), para penalizar tiros que só funcionam no ângulo exato.
4. **Checkpoint:** nova sessão; métrica principal = dano real / previsto.
5. **`chore/pendencias`:** restart via `ponto_hud`, remover `--debug`/`--smart-wait`,
   `sys.path` em `tools/snapshot.py`/`debug_view.py`, tirar este arquivo do git
   (`git rm --cached`).
6. Depois: `calibrar.py` automático (acha a borda do tabuleiro e grava `calibracao.json`).

Backlog que continua valendo:

- **Disputa de `mss` entre threads:** o painel "bolas" das janelas de debug chama
  `read_ball_count_template` (mss global de `vision.py`) da thread de debug — e agora
  também mexe em `CONTADOR_CANDIDATOS`. O painel vem desligado (`JANELAS["bolas"]`);
  não ligar durante uma partida até isso ser resolvido.
- **`historico.txt` grava `Bolas: 0`:** o número é lido na tela de game over.
- **`--dry-run` não simula:** só escaneia e alimenta as janelas de debug.
- **Templates do contador:** refazer no tamanho nativo (~13 px) usando os
  `turno_NNN_contador.png` rotulados pelo filtro. Uma comparação em tons de cinza foi
  testada e descartada (leu "x61" como 7 ao vivo).
- **Testes de regressão** da lógica pura (física, grade, filtros).

---

## Recordes (`historico.txt`)

- 2026-06-23 — **63 turnos** (recorde)
- 2026-06-17 — 54 turnos
- 2026-06-15 — 52 turnos
- 2026-09-26 — 51 turnos
- 2026-06-24 — 51 turnos

A coluna `Bolas` não é confiável (antes gravava sempre 1; hoje grava 0).
