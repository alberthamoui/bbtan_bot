# SITUAÇÃO ATUAL DO PROJETO — BBTAN Bot

Bot em Python que joga o BBTAN (brick-breaker mobile) de forma autônoma, rodando
num emulador Android no Windows. Usa visão computacional (OpenCV + mss) para ler
o estado do jogo e PyAutoGUI para executar os disparos.

- **Branch:** `main`
- **Último commit refletido:** `df96afa style: aplicar formatação automática (Black)`
- **Data desta atualização:** 2026-07-26

> Esta revisão foi escrita após mapear o código real (config, vision, physics,
> executor, main, main_vision, monitor, tools). A versão anterior deste documento
> refletia o commit `87f2f33` (2026-07-01) e ficou defasada em vários pontos —
> ver a seção **[Divergências corrigidas nesta revisão](#divergências-corrigidas-nesta-revisão)**.

---

## Resumo executivo — status dos módulos

| Módulo | Papel | Status |
|--------|-------|--------|
| `config.py` | Constantes, HSV, regiões, pesos | ✅ Estável. `BALL_RADIUS` agora fixo em **9 px** (era `BLOCK_SIZE*0.50=32`). Tem lixo conhecido: `BLOCK_COLOR_RANGES` com entradas duplicadas. |
| `vision.py` | Captura + detecção (blocos / power-ups / bola / HP) | ✅ Funcional. Pipeline 2x2/triângulo/HP maduro. `_read_round_number` **removido**. Riscos abertos: race no `mss`, sem dedup power-up↔bloco, OCR de HP sem clamp. |
| `physics.py` | Simulação de trajetória + escolha de ângulo | ✅ Funcional. Agora grava **path por bola** (`ball_paths`/`ball_collisions`) para animação; batch + laser + power-ups. |
| `executor.py` | Disparo via PyAutoGUI | ✅ Funcional. **Só `execute_shot_swipe`** sobrou; `tap`/`hold_release` removidos. `wait_for_turn_end*` **removidos** junto com `TURN_WAIT_TIME`. |
| `main.py` | Loop principal | ✅ Funcional. Bugs menores conhecidos: `q` no `--debug` retorna tupla; `--dry-run` não roda decisão; coords de restart/unstick hardcoded. |
| `main_vision.py` | Loop só de visão | ✅ Ferramenta de validação, sem física/mouse. |
| `monitor.py` | Monitor lado-a-lado (detecção + sim) | ✅ Ferramenta de diagnóstico. |
| `tools/*` | Calibração, captura de templates, animação, masks | ✅ Todos movidos/normalizados para `tools/`. |
| Testes | — | ❌ **Inexistentes.** Lógica pura (física, grid, OCR) 100% testável mas sem cobertura. |

**Maiores riscos abertos hoje:** (1) qualidade da detecção de HP (sem clamp/sanidade),
(2) race condition da instância `mss` compartilhada entre threads, (3) ausência
total de testes de regressão — a lógica de visão já regrediu silenciosamente antes.

---

## Visão geral do fluxo (1 turno)

1. `main.py` → `time.sleep(3)` → `scan_game()` (vision) captura o frame da
   `GAME_REGION`, detecta blocos/power-ups/bola, lê contagem de bolas e testa
   `game_over` (>25% de pixels claros).
2. `update_level_display(...)` alimenta as janelas de debug (thread daemon).
3. `run_turn(state)` chama `find_best_angle(...)` de `physics.py`, que simula
   TODOS os ângulos `ANGLE_MIN..ANGLE_MAX` (19..161) em passo de 1°, ranqueia e
   re-simula o melhor com `record_path=True`.
4. `executor.execute_shot(ball_x, ball_y, angle)` faz `moveTo` + `drag` na tela e
   clica no botão de acelerar (`SPEED_UP_X, SPEED_UP_Y`).
5. Poll de `detect_round_end()` a cada 0.3 s (timeout 30 s) — quando ≥2 de 4
   regiões-indicadoras rosa acendem, a rodada acabou.
6. `time.sleep(2.0)` (blocos descem/estabilizam). Volta para 1.

Detecta jogo travado (fingerprint dos blocos idêntico por 20 turnos consecutivos)
e clica `(1580, 961)` para destravar. Em game over: loga em `historico.txt`,
executa 3× a sequência de restart (`(1576,781)` e `(1584,838)`) e reseta o estado
interno.

---

## Estrutura de arquivos

```
bbtan_bot/
├── config.py                    — Constantes e parâmetros
├── vision.py                    — Captura + detecção (blocos / power-ups / bola / HP)
├── physics.py                   — Simulação de trajetória + escolha de ângulo
├── executor.py                  — Disparo via PyAutoGUI
├── main.py                      — Loop principal (--debug, --dry-run, --animate, --verbose-vision...)
├── main_vision.py               — Loop SÓ de visão (sem física/disparo)
├── monitor.py                   — Monitor lado-a-lado (detecção + simulação)
├── historico.txt                — Log de partidas (turnos sobrevividos + bolas)
├── README.md                    — Setup/uso (⚠️ desatualizado: aponta arquivos na raiz que hoje vivem em tools/)
├── requirements.txt             — Dependências (opencv-python, numpy, mss, pyautogui)
├── docs/
│   ├── situacao_atual.md        — ESTE documento
│   ├── plano_melhorias.md       — Backlog analítico de bugs/melhorias (companheiro deste doc)
│   └── notes.md                 — Rascunho de bugs / cores / calibração
├── templates/
│   ├── digits/                  — Templates 28px do BALL_COUNT (0.png..9.png)
│   └── hp_digits/               — Templates 20px de HP (0.png..9.png) +
│                                  ~140 amostras brutas <digito>_<ts>.png
└── tools/
    ├── animate_dryrun.py        — Dry-run animado (leque de ângulos + bola percorrendo o path)
    ├── visualizar_masks.py      — Thread com janelas de debug ao vivo
    ├── debug_view.py            — Overlays da janela de debug do main
    ├── calibrate.py             — Calibração de GAME_REGION
    ├── calibrate_levels.py      — Calibração de _GRID_TOP_Y / fileiras
    ├── calibrate_columns.py     — Calibração de COL_LEFTS / COL_RIGHTS
    ├── calibrate_powerup.py     — Medição de raio de power-ups
    ├── calibrate_digits.py      — Calibração de templates do BALL_COUNT
    ├── calibrate_hp_digits.py   — Calibração de templates de HP
    ├── capture_hp_templates.py  — Captura interativa de amostras de dígitos HP
    ├── consolidate_hp_templates.py — Consolida amostras em templates médios 20px
    ├── seletor_hsv.py           — Ajuste interativo de faixas HSV
    ├── get_area.py              — Captura de região
    ├── get_position.py          — Lê posição do mouse
    ├── snapshot.py              — Screenshot do jogo
    ├── masks.py                 — Utilitários de mask
    ├── grid_diagnostic.py       — Diagnóstico visual da grade
    └── test_tesseract.py        — Sandbox de OCR (não usado no pipeline ativo)
```

> **Nota:** `capture_hp_templates.py`, `consolidate_hp_templates.py` e utilitários
> de calibração agora vivem em `tools/` (antes estavam na raiz). Nomes normalizados
> para `snake_case` (`get_area.py`, `get_position.py`, `test_tesseract.py`).

---

## config.py — parâmetros principais

### Região do jogo

```python
GAME_REGION = (1352, 244, 452, 578)        # antigo: (1353, 245, 451, 577)
WALL_LEFT   = 0                             # antigo: 5
WALL_RIGHT  = 449                           # antigo: GAME_REGION[2]-5 (=446)
WALL_TOP    = 2                             # antigo: 5
WALL_BOTTOM = 575                           # antigo: GAME_REGION[3]-10 (=567)
GRID_COLUMNS = 7
BLOCK_SIZE   = GAME_REGION[2] // GRID_COLUMNS  # 64 px por bloco
```

### Grid

```python
COL_LEFTS  = [4, 67, 132, 195, 258, 321, 386]
COL_RIGHTS = [63, 128, 191, 255, 318, 382, 444]

_GRID_TOP_Y = 67
LEVEL_TOPS = [_GRID_TOP_Y + (6 - i) * BLOCK_SIZE for i in range(7)]
LEVEL_BOTS = [_GRID_TOP_Y + (7 - i) * BLOCK_SIZE - 1 for i in range(7)]
LEVEL_BOTS[0] = 513   # ajuste manual: bater com a base do bloco mais baixo
LEVELS = [(t + b) // 2 for t, b in zip(LEVEL_TOPS, LEVEL_BOTS)]
```

- `row_index 0` = fileira MAIS BAIXA (CRÍTICA — display L7).
- `row_index 6` = fileira do TOPO (display L1, onde blocos novos spawnam).
- Convenção de display ao usuário: `Level = 7 - row_index`.

### HSV — blocos e HP

```python
BLOCK_HSV_LOW  = np.array([0, 60, 60])     # captura também blocos escuros (roxo)
BLOCK_HSV_HIGH = np.array([179, 255, 255])
HP_TEXT_HSV_LOW  = np.array([0, 0, 0])      # texto escuro sobre fundo colorido
HP_TEXT_HSV_HIGH = np.array([179, 255, 130])  # antigo: V<=99 (só valia no fallback)
MIN_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 0.15)
MAX_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 4.5)
```

`BLOCK_COLOR_RANGES` (config.py:70) lista `(nome, low, high)` por cor de bloco.
⚠️ **Contém redundância conhecida:** `amarelo_ouro` aparece 2× e `laranja2` é
idêntico a `laranja` — infla o denominador de dominância em `_qualifies_as_big_block`
(ver To-Do B7). Usada para:
- Isolar o texto do HP dentro do hull colorido (`_get_block_text_mask`).
- Validar se um contorno 2x2 é um bloco grande real (uma cor > 40% da ROI).
- Gate `_qualifies_as_big_block` (dominância ≥ 80% da área mascarada).

### HSV — power-ups

```python
POWERUP_LASER_HSV_LOW/HIGH       = azul  #008beb   (H 97..107)
POWERUP_EXTRA_BALL_HSV_LOW/HIGH  = ciano #00edd3   (H 81..91)
POWERUP_B_HSV_LOW/HIGH           = ouro  #ffd539   (H 20..26)
```

### Bola  ⚠️ mudança relevante

```python
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.50))  # antigo: 32 px (inflava colisões)
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.22))  # tentativa intermediária
BALL_RADIUS   = 9        # valor atual (fixo)
BALL_SPEED    = 8.0      # ⚠️ DECORATIVO — se anula algebricamente (ver abaixo)

SIM_PATH_BUDGET = 4000.0                            # px de percurso por voo
SIM_STEP_SIZE   = 1.0                               # antigo: 0.5, antes 2.0
MAX_SIM_STEPS   = int(SIM_PATH_BUDGET / SIM_STEP_SIZE)
```

O `BALL_RADIUS = 32` que dominava o backlog anterior **já foi reduzido para 9**
(commit `6b46855`). O ponto em aberto agora é revalidar `count_balls` e o overlay
de animação contra o novo raio.

**`BALL_SPEED` é decorativo.** `_sim_one_ball` faz `vel.normalized() * BALL_SPEED`
e depois avança `vel * (SIM_STEP_SIZE / BALL_SPEED)` — multiplica e divide pelo
mesmo valor. O avanço real é sempre `direção × SIM_STEP_SIZE`, qualquer que seja o
`BALL_SPEED`. Mantido só para não quebrar imports.

**`MAX_SIM_STEPS` × `SIM_STEP_SIZE` são acoplados.** O que governa o alcance da
bola é o PRODUTO dos dois. Isso não estava documentado, e um ajuste de precisão
(`SIM_STEP_SIZE` 2.0 → 0.5 com `MAX_SIM_STEPS` 2000 → 3000) derrubou o orçamento de
4000 px para 1500 px sem que nada avisasse. Como o playfield tem 573 px de altura e
subir+voltar a 19° custa 3520 px, **51 dos 143 ângulos (36%) eram truncados no meio
do voo** — justamente os rasos, que são os que mais quicam e destroem.

Agora o orçamento é declarado (`SIM_PATH_BUDGET`) e `MAX_SIM_STEPS` é derivado dele:
`SIM_STEP_SIZE` controla só precisão, `SIM_PATH_BUDGET` só alcance. Medido após a
correção: **1/143 ângulos truncados**, a 4.01 s por turno (contra 5.87 s do
`3000 × 0.5`, que tinha 2.7× menos alcance).

### Detecção de fim de rodada (4 regiões absolutas com cor rosa)

```python
ROUND_END_REGIONS = [
    (1445, 920, 22, 29), (1544, 919, 25, 30),
    (1649, 922, 22, 28), (1748, 919, 27, 32),
]
ROUND_END_HSV_LOW  = np.array([163, 150, 180])   # rosa #e53572
ROUND_END_HSV_HIGH = np.array([177, 255, 255])
```

Cada região "hita" se >10% dos pixels caem na faixa. Rodada acabou quando ≥2 das 4 hitam.

### Scoring por fileira

```python
ROW_MULTIPLIERS = {
    0: 100.0,  # GAME OVER IMINENTE (L7)
    1: 50.0,   # PERIGO (L6)
    2: 2.5,  3: 1.5,  4: 1.0,  5: 0.8,  6: 0.5,   # L1 (topo)
}
W_HIT          = 1.0    # por hit, já ponderado por ROW_MULTIPLIERS
W_DESTROY      = 5.0    # por bloco destruído, já ponderado por ROW_MULTIPLIERS
W_EXTRA_BALL   = 30.0   # por bola extra coletada
W_BONUS_B      = 5.0    # por moeda B coletada
W_ROW0_SURVIVE = 0.0    # DESLIGADO — por HP vivo na fileira crítica ao fim do turno
W_LOOP         = 0.0    # DESLIGADO — por grupo de 10 hits no mesmo voo
```

Ranking em `find_best_angle`: **ordenação única por `(score, total_hits)` desc**,
onde `score = physics.compute_score(ScoreTerms)`.

#### Score unificado — o que mudou e por quê

Antes existiam **duas notas paralelas**: `score` (alimentado pelos `SCORE_*`) e
`weighted_hits` (alimentado pelos `*_WEIGHT`). O ranking usava só a segunda, então
cinco parâmetros do config eram decorativos — só apareciam no print do console:

| antigo | virou | observação |
|---|---|---|
| `SCORE_HIT_BLOCK = 1` | `W_HIT` | — |
| `SCORE_DESTROY_BLOCK = 50` | — | removido: redundante com `W_DESTROY` |
| `SCORE_HIT_POWERUP = 200` | — | removido: redundante com `W_EXTRA_BALL`/`W_BONUS_B` |
| `SCORE_LOOP_BONUS = 100` | `W_LOOP` | agora efetivo, começa em 0 |
| `SCORE_ROW0_SURVIVE_PENALTY = 100` | `W_ROW0_SURVIVE` | agora efetivo, começa em 0 |
| `DESTROY_WEIGHT = 5` | `W_DESTROY` | só renomeado |
| `EXTRA_BALL_WEIGHT = 30` | `W_EXTRA_BALL` | só renomeado |
| `BONUS_B_WEIGHT = 5` | `W_BONUS_B` | só renomeado |

`W_ROW0_SURVIVE` merece destaque: é a **única regra do projeto que olha o estado
final do tabuleiro** ("não deixe bloco vivo na fileira que causa game over"), e
nunca foi aplicada uma única vez. Entra desligada, para ser ligada e medida sozinha.

**A camada `danger_hits` caiu do ranking.** Ela ordenava lexicograficamente por
hits nas rows 0-1 antes de qualquer outra coisa, o que fazia um tiro que belisca 3×
um bloco de 50 HP da fileira crítica vencer um que a destrói e ainda limpa 5 outros.
Os `ROW_MULTIPLIERS` (row 0 = 100×) já codificam esse perigo de forma contínua.
`danger_hits` continua no `SimResult` e no console, como diagnóstico.

**Equivalência verificada:** com `W_LOOP=0` e `W_ROW0_SURVIVE=0`, o `score` novo é
numericamente idêntico ao `weighted_hits` antigo (erro exatamente 0 em 18 tabuleiros
sintéticos × 143 ângulos). Nesses cenários o ângulo escolhido mudou em **2 de 18**
casos, ambos pela queda do `danger_hits` — inclusive um salto grande (155° → 27°),
esperado quando o desempate lexicográfico deixa de forçar a escolha.

### Ângulos e execução  ⚠️ faixa de ângulos mudou

```python
ANGLE_MIN  = 19        # antigo: 10
ANGLE_MAX  = 161       # antigo: 170  (simétrico: 180 - 19)
ANGLE_STEP = 1         # ⚠️ DECORATIVO — find_best_angle usa passo 1 chumbado
BALL_COUNT_REGION   = (1308, 824, 536, 33)
BALL_COUNT_HSV_LOW  = np.array([0, 0, 140])
BALL_COUNT_HSV_HIGH = np.array([179, 50, 255])
SPEED_UP_X, SPEED_UP_Y = 1789, 958
SPEED_UP_DELAY = 0.3
DRAG_DISTANCE  = 150
DRAG_DURATION  = 0.15
VISION_VERBOSE = False   # ligado por --verbose-vision (NOVO)
```

---

## vision.py — detecção

### Estruturas

- `Block` (dataclass): `x, y, w, h, cx, cy, block_type, hp, vertices,
  tri_orientation, row_index, col_index, hp_mask`. `hp_mask` guarda a `text_mask`
  gerada por `_get_block_text_mask` — alimenta a janela `DEBUG_MASK_HP`.
- `PowerUp`: `cx, cy, radius, power_up_type`. `PowerUpType` =
  `EXTRA_BALL | LASER_H | LASER_V | BONUS_B | UNKNOWN`.
- `BlockType` = `SQUARE | TRIANGLE | UNKNOWN`; `TriangleOrientation` = 4 cantos +
  `UNKNOWN`.
- `GameState`: blocos, power-ups, bola, `detected_ball_count`, `game_over`,
  `white_ratio`, `raw_frame`, `debug_frame`.

### Leitura do número de bolas — `read_ball_count_template()`

Template matching em `BALL_COUNT_REGION` contra `templates/digits/` (28px, gerado
com `calibrate_digits.py`). Filtra pelo HSV claro, pula o primeiro componente
conectado (o "x") e concatena os dígitos restantes. Fallback: `count_balls(frame)`
— contagem de blobs claros na base do frame.

### Pipeline de blocos — `detect_blocks(frame)`

1. HSV + `cv2.inRange(BLOCK_HSV_LOW..HIGH)` → `raw_mask`.
2. Morfologia (CLOSE/OPEN) → `mask`.
3. `findContours(RETR_EXTERNAL)`.
4. Por contorno: descarta pequenos (`area < MIN_BLOCK_AREA` ou `w<35 / h<35`),
   faz **big-area split** (>`MAX_BLOCK_AREA`) ou mapeia na grid.

### `_qualifies_as_big_block(hsv, x, y, w, h)` — gate 3-checks

Filtro comum usado tanto no caminho normal 2x2 quanto no `big-area phase 2`:
1. **Dimensão**: `w`/`h` em `2*BLOCK_SIZE ± 15%`.
2. **Cor uniforme dominante**: a cor top deve cobrir ≥ 80% do total mascarado.
3. **Contagem de HP-clusters**: um 2x2 legítimo tem UM cluster (o HP central);
   2+ clusters ⇒ rejeita.

Rejeições logadas com `[big-block reject ...]` **apenas se `--verbose-vision`**
(prints da visão agora ficam atrás do flag `cfg.VISION_VERBOSE`, commit `117ad4f`).

### Big-area split (3 fases)

1. **Fase 1 — survivors**: exige `fill ≥ 0.40` por célula; classifica cada
   sobrevivente como SQUARE/TRIANGLE.
2. **Fase 2 — 2x2 por cor uniforme + gate**: emite UM bloco 2x2 quando as 4
   células sobrevivem, alguma cor cobre >40% e `_qualifies_as_big_block` passa.
   HP lido com `is_2x2=True`.
3. **Fase 3 — split em 1x1**: células não consumidas viram blocos individuais.

### Caminho normal — 1x1, 2x2 candidato, fundido

- **1x1**: classifica via `raw_mask` (`hull_fill < 0.60` ⇒ triângulo).
- **2x2 candidato**: gate de cor uniforme + `_qualifies_as_big_block`. Se passa,
  emite 1 bloco 2x2 **lendo HP com `is_2x2=True`** (corrigido no commit `c988452`;
  antes o caminho normal esquecia essa flag e cortava o HP de 3 dígitos). Se falha,
  split em 1x1 com `fill > 15%` por célula.
- **Fundido (2x1, 1x2)**: split por célula com `fill > 15%`.

### Atribuição de HP — `_read_hp(...)`

OCR puro contra `templates/hp_digits/<digit>.png` (20px), sem tracking entre frames.
- `_get_block_text_mask`: usa o hull da cor dominante como `block_region` (quadrados)
  ou a ROI inteira (triângulos, para não cortar o dígito na hipotenusa).
  `dark = inRange(v_channel, 0, 130)`; `text_mask = dark & block_region`.
- Padding e `min_area` ajustados por caso (`is_2x2`, triângulo, 1x1).
- Se `text_mask` < 5 px ou nenhum componente válido: retorna `hp=1`.
- Score por template = fração de pixels iguais; threshold 0.6.
- ⚠️ **Sem clamp de sanidade**: `int(digits_str)` pode anexar ruído (ver To-Do B5).

### Power-ups / bola / game over / fim de rodada

- `detect_powerups`: HSV + circularidade > 0.6 + raio 7..15 px. `LASER_H/V`
  distinguido por projeção linha/coluna.
- `detect_game_over`: pixels claros > 25% do frame.
- `detect_round_end`: cobertura rosa >10% em ≥2 das 4 `ROUND_END_REGIONS`.

### `_read_round_number` — REMOVIDA

Era órfã (template matching de round number, sobra da época em que alimentava HP).
Removida no cleanup `7a2cd06`. HP hoje é OCR puro contra `templates/hp_digits/`.

---

## physics.py — simulação

### `SimResult`

```python
angle: float
score: float               # display apenas
total_hits: int
weighted_hits: float       # hits × ROW_MULTIPLIERS (diagnóstico)
danger_hits: int           # hits efetivos em rows 0-1 (diagnóstico, fora do ranking)
blocks_destroyed: int
powerups_collected: int
extra_balls_collected: int
path: List[Tuple]          # path da 1a bola (só com record_path=True)
collisions: List[Tuple]
ball_paths: List[List[Tuple]]        # NOVO: um path POR bola lançada
ball_collisions: List[List[Tuple]]   # NOVO: colisões POR bola
hit_details: List[str]
```

> **Mudança "solo → todas as bolas" (commit `6b46855`):** com `record_path=True`,
> a simulação agora grava a trajetória **de cada bola** (`ball_paths`/`ball_collisions`),
> não só a da primeira. Bolas posteriores seguem trajetórias re-simuladas após
> blocos destruídos — usado pelo `animate_dryrun` para não repetir o path da 1a bola.

### `_sim_one_ball`

Simulação de uma bola. Colisões: AABB (quadrados), triângulos (cateto vs hipotenusa
por dot product com a normal diagonal), power-ups (circular, coletados pós-bola).
`record_path=True` amostra path a cada 4 steps e registra colisões.

### `simulate_trajectory` — loop de bolas com batch

- **Power-ups**: `LASER_H/V` aplica `_apply_laser` (1 de dano por bola do batch na
  linha/coluna) e força `batch = 1`; `EXTRA_BALL` incrementa `terms.extra_balls`;
  `BONUS_B` incrementa `terms.bonus_b` (os pesos são aplicados só em `compute_score`).
- **Batch**: agrupa bolas de mesma trajetória até destruir o próximo bloco. Forçado
  a 1 se laser disparou **ou** se `record_path=True` (para gravar path exato de cada bola).
- Pós-loop: soma em `terms.row0_surviving_hp` o HP de cada bloco de row 0 que
  sobreviveu. Os pesos (`W_ROW0_SURVIVE`, `W_LOOP`) entram só em `compute_score`.

### `find_best_angle`

Simula 1° em 1° (19..161). Ordena conforme presença de blocos em row 0-1. Re-simula
o melhor ângulo com `record_path=True`.

---

## main.py — loop principal

### Flags

```
python main.py                  # bot normal
python main.py --debug          # janela de debug com trajetória
python main.py --dry-run        # visualização contínua (⚠️ hoje NÃO roda a decisão, só escaneia)
python main.py --animate        # dispatch tools/animate_dryrun.run_animate()
python main.py --delay N        # delay inicial em segundos (default 1.0)
python main.py --max-turns N    # encerra após N turnos
python main.py --smart-wait     # ⚠️ flag parseada mas NÃO usada pelo loop
python main.py --verbose-vision # habilita prints de debug do módulo vision (NOVO)
```

> `--balls N` **não existe mais** como flag no argparse atual (a versão anterior do
> doc listava). O `ball_count` vem de `state.detected_ball_count` (mín. 1).

### Detecção de jogo travado

`fingerprint = frozenset((row_index, round(cx / BLOCK_SIZE)) for b in blocks)`. Se
idêntico por 20 turnos, clica `(1580, 961)` para destravar.

### Game over

Loga em `historico.txt` (`YYYY-MM-DD HH:MM:SS | Turnos: N | Bolas: M`), executa 3
sequências de clique em `(1576, 781)` e `(1584, 838)`, reseta estado e recomeça.
⚠️ Coordenadas de restart/unstick são números mágicos no `main.py` (deveriam estar
em `config.py`).

### Thread de debug — `tools/visualizar_masks.py:run_masks_window`

Daemon thread com as janelas: "Blocos + PowerUps", "Qtd Bolas", "Fim de Rodada",
"Blocos por Level" e "DEBUG_MASK_HP" (mosaico 7x7 das `text_mask` + HP lido).
⚠️ Chama `read_ball_count_template()` que usa o `mss` global compartilhado com a
thread principal (race condition — ver To-Do B3).

---

## executor.py — disparo

- `angle_to_screen_coords`: converte `(ball_x, ball_y, angle_deg)` + `DRAG_DISTANCE`
  em coords absolutas de tela.
- **Único método ativo:** `execute_shot_swipe` (`moveTo` + `drag`), aliased como
  `_base_shot`. `execute_shot` faz o swipe + `speed_up_click()`.
  > `execute_shot_tap` e `execute_shot_hold_release` **foram removidos** — o README
  > ainda os menciona (desatualizado).
- `wait_for_turn_end` / `wait_for_turn_end_smart` **foram removidos** — nunca eram
  chamados (o loop usa `detect_round_end`). `cfg.TURN_WAIT_TIME` saiu com eles.
- `pyautogui.FAILSAFE = True` (canto sup. esq. aborta).

---

## main_vision.py / monitor.py / animate_dryrun.py

- **`main_vision.py`**: roda `scan_game()` em loop, imprime no console e mantém a
  thread de masks viva. Sem física/mouse. `python main_vision.py [--delay N] [--interval S]`.
- **`monitor.py`**: janela única, painel esquerdo (detecção) + direito (top-5 da
  simulação). `python monitor.py [--balls N] [--fps N] [--no-sim]`.
- **`tools/animate_dryrun.py`**: Phase 1 (leque de todos os ângulos) + Phase 2 (bolas
  percorrendo `ball_paths`). Invocado por `python main.py --animate`.

---

## Templates de HP — pipeline de captura (em `tools/`)

1. **`tools/capture_hp_templates.py`** — loop interativo: mostra cada componente da
   `hp_mask` ampliado 10x; teclas `0-9` salvam o crop como
   `templates/hp_digits/<dígito>_<timestamp>.png`.
2. **`tools/consolidate_hp_templates.py`** — para cada dígito 0..9: lê amostras,
   redimensiona para altura 20 px, tira a média + threshold, salva
   `templates/hp_digits/<dígito>.png`.

`_load_hp_digit_templates` filtra por nome de 1 caractere, então as ~140 amostras
brutas na pasta não atrapalham o runtime.

---

## Dependências (`requirements.txt`)

```
opencv-python
numpy
mss
pyautogui
```

---

## Divergências corrigidas nesta revisão

Ajustes feitos para alinhar o documento ao código real (pós-`87f2f33`):

1. **`BALL_RADIUS`** 32 px → **9** (hardcoded). Era o item nº1 do backlog antigo.
2. **Ângulos** `10..170` → **`19..161`** (`ANGLE_MIN=19`, `ANGLE_MAX=161`).
3. **`_read_round_number` removida** do `vision.py` (não é mais "órfã presente").
4. **`execute_shot_tap` / `execute_shot_hold_release` removidos** — só o swipe sobrou.
5. **`SimResult`** ganhou `ball_paths` / `ball_collisions` (path por bola).
6. **`BONUS_B_WEIGHT = 5`** adicionado; **`SCORE_MULTI_BOUNCE` removido** do config.
7. **Bug do 2x2 no caminho normal** (esquecia `is_2x2=True`) **corrigido** (`c988452`).
8. **Prints de debug da visão** agora atrás de `--verbose-vision` / `cfg.VISION_VERBOSE`.
9. **Arquivos movidos para `tools/`** e renomeados para `snake_case`; `notes.md`
   está em `docs/`, não na raiz.
10. **Flag `--balls`** não existe mais no argparse do `main.py`.

---

## To-Do priorizado

Derivado de `docs/plano_melhorias.md` + `docs/notes.md`, já descontando o que foi
resolvido (BALL_RADIUS, is_2x2 do 2x2). Ordem sugerida:

### Alta prioridade

1. **[B3] Race condition do `mss` compartilhado.** `vision._sct` é uma instância
   global usada pela thread principal (`scan_game`/`detect_round_end`) e pela thread
   de masks (`read_ball_count_template`). `mss` não é thread-safe → frames corrompidos
   / `ScreenShotError` intermitentes. Corrigir com `mss` thread-local ou lock nos grabs.
2. **[D1] Testes de regressão da lógica pura.** Zero testes hoje. `physics` (colisões,
   batch, laser), `_get_row_index/_get_col_index`, clustering de `_qualifies_as_big_block`
   e OCR de HP contra os PNGs de amostra são determinísticos e testáveis offline. A
   lógica de visão já regrediu silenciosamente antes.

### Média prioridade

3. **[B5] Hardening do OCR de HP.** `_read_hp` faz `int(digits_str)` sem clamp
   (ruído anexa dígitos: "1" → "17"); fallback de cobertura roda HSV na ROI inteira.
   Clampar (máx. 3 dígitos + range plausível), restringir busca ao centro no fallback,
   descartar componentes com altura muito fora da mediana. Erros aqui envenenam o batch.
4. **[B4] Dedup power-up ↔ bloco.** `detect_blocks` capta power-ups pelo HSV amplo;
   única proteção é `w<35 or h<35`. Cross-check com `detect_powerups` para as bolas
   nunca quicarem em posição de power-up na simulação.
5. **[C1] `q` no `--debug` retorna tupla.** `run_turn` faz `return False, 0` (truthy);
   `main()` trata como fingerprint e espera 30 s por um fim de rodada que não vem.
   Retornar status "quit" distinto e sair limpo do loop.
6. **[C2] `historico.txt` sempre grava `Bolas: 1`.** `ball_count` local nunca
   sincroniza com `state.detected_ball_count` → a métrica de acompanhamento está
   errada. Sincronizar após cada turno bem-sucedido (>0).

### Baixa prioridade

7. **[B7] Limpeza.** Dedupe `BLOCK_COLOR_RANGES` (`amarelo_ouro` 2×, `laranja2 == laranja`
   — inflam o denominador de dominância).
8. **[C3] `--dry-run` não roda a decisão.** O branch só escaneia; nunca chama
   `run_turn(dry_run=True)`. Fazer rodar o pipeline de decisão + overlay do `best.path`.
9. **[C4] Flag/coords mortas.** ~~Remover `wait_for_turn_end*`~~ (feito). Remover `--smart-wait`; mover
   coords de restart/unstick para `config.py` (quebram na próxima recalibração).
10. **[C5] Sleeps fixos.** Substituir `sleep(3)` pré-scan e `sleep(2.0)` pós-rodada
    por checagem de estabilidade (fingerprint idêntico em 2 scans ~0.3 s).
11. **Revalidar `count_balls` e overlay de animação** contra o novo `BALL_RADIUS=9`.
12. Atualizar o **README.md** (aponta arquivos na raiz que hoje estão em `tools/` e
    métodos de disparo removidos).

---

## Pontos conhecidos em aberto (`docs/notes.md`)

- **`count_balls` usa `BALL_RADIUS` para estimar área** — revalidar com raio 9.
- **`detect_blocks` capta power-ups como blocos** pelo HSV amplo; mitigado por
  `w<35 or h<35`, sem dedup explícita (To-Do B4).
- **`historico.txt` grava `Bolas: 1` fixo** (To-Do C2). Não afeta gameplay porque
  `run_turn` usa `state.detected_ball_count`.
- **Sem tracking entre frames**: cada frame é detectado do zero. Erros de OCR não se
  propagam, mas o bot perde info quando um bloco "pisca".

---

## Recordes recentes (`historico.txt`)

- 2026-06-23 23:39:54 — **63 turnos** | RECORDE
- 2026-06-17 00:40:05 — 54 turnos
- 2026-06-15 01:02:55 — 52 turnos
- 2026-06-24 00:01:17 — 51 turnos
- 2026-06-17 00:56:04 — 49 turnos
- 2026-07-01 22:52:06 — 42 turnos (primeira partida pós-recalibração de grid)

Todas as linhas do log gravam `Bolas: 1` por causa do bug C2 — a coluna de bolas
não é sinal confiável até que seja corrigida.
