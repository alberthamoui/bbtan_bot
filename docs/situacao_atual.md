# SITUACAO ATUAL DO PROJETO — BBTAN Bot

Bot em Python que joga o BBTAN (brick-breaker mobile game) de forma autonoma,
rodando em emulador Android no Windows. Usa visao computacional (OpenCV + mss)
para ler o estado do jogo e pyautogui para executar os disparos.

Ultimo commit relevante: `87f2f33 feat(vision): 2x2 big-block gate + grid/wall recalibration`.
Este documento reflete o estado do codigo pos-commit (`main` em 2026-07-01).

---

## Visao geral do fluxo (1 turno)

1. `main.py` -> `time.sleep(3)` -> `scan_game()` (vision) captura frame da regiao
   do jogo, detecta blocos/power-ups/bola, le contagem de bolas e testa
   `game_over` (25% de pixels brancos).
2. `update_level_display(...)` alimenta as janelas de debug (thread separada).
3. `run_turn(state)` chama `find_best_angle(...)` de `physics.py`, que simula
   TODOS os angulos 10..170 em passo de 1 grau, ranqueia e re-simula o melhor
   com `record_path=True`.
4. `executor.execute_shot(ball_x, ball_y, angle)` faz `moveTo` + `drag` na tela
   e clica no botao de acelerar (`SPEED_UP_X, SPEED_UP_Y`).
5. Poll de `detect_round_end()` a cada 0.3s (timeout 30s) — quando 2 de 4
   regioes-indicadoras rosa acendem, a rodada acabou.
6. `time.sleep(2.0)` (blocos descem/estabilizam). Volta para 1.

Detecta jogo travado (fingerprint dos blocos identico por 20 turnos consecutivos)
e clica `(1580, 961)` para destravar. Em game over, loga em `historico.txt`,
executa 3x a sequencia de restart (`(1576,781)` e `(1584,838)`) e reseta o
estado interno.

---

## Estrutura de arquivos

```
bbtan_bot/
├── config.py                    — Constantes e parametros
├── vision.py                    — Captura + deteccao (blocos / power-ups / bola / HP)
├── physics.py                   — Simulacao de trajetoria + escolha de angulo
├── executor.py                  — Disparo via pyautogui
├── main.py                      — Loop principal (com --debug, --dry-run, --animate...)
├── main_vision.py               — Loop SO de visao (sem fisica/disparo)
├── monitor.py                   — Monitor lado-a-lado (deteccao + simulacao)
├── capture_hp_templates.py      — Captura interativa de amostras de digitos HP
├── consolidate_hp_templates.py  — Consolida amostras em templates medios 20px
├── historico.txt                — Log de partidas (turnos sobrevividos + bolas)
├── notes.md                     — Rascunho de bugs / cores / calibracao
├── templates/
│   ├── digits/                  — Templates 28px do BALL_COUNT (`0.png..9.png`)
│   └── hp_digits/               — Templates 20px de HP (`0.png..9.png`) +
│                                  amostras brutas `<digit>_<ts>.png`
└── tools/
    ├── animate_dryrun.py        — Visualizacao animada do dry-run (Phase 1 fan
    │                              de angulos + Phase 2 bola percorrendo o path)
    ├── visualizar_masks.py      — Thread com 4-5 janelas de debug ao vivo
    ├── debug_view.py            — Overlays da janela de debug do main
    ├── calibrate.py             — Calibracao de GAME_REGION
    ├── calibrate_levels.py      — Calibracao de _GRID_TOP_Y / fileiras
    ├── calibrate_columns.py     — Calibracao de COL_LEFTS / COL_RIGHTS
    ├── calibrate_powerup.py     — Medicao de raio de power-ups
    ├── calibrate_digits.py      — Calibracao de templates do BALL_COUNT
    ├── calibrate_hp_digits.py   — Calibracao de templates de HP
    ├── seletor_hsv.py           — Ajuste interativo de faixas HSV
    ├── getArea.py               — Captura de regiao
    ├── getPosition.py           — Le posicao do mouse
    ├── snapshot.py              — Screenshot do jogo
    ├── masks.py                 — Utilitarios de mask
    ├── grid_diagnostic.py       — Diagnostico visual da grade
    └── testTesseract.py         — Sandbox de OCR (nao usado no pipeline ativo)
```

---

## config.py — parametros principais

### Regiao do jogo (recalibrada)

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
LEVEL_BOTS[0] = 513   # ajuste manual: bater exato com a base do bloco mais baixo
LEVELS = [(t + b) // 2 for t, b in zip(LEVEL_TOPS, LEVEL_BOTS)]
```

- `row_index 0` = fileira MAIS BAIXA (CRITICA — display L7).
- `row_index 6` = fileira do TOPO (display L1, onde blocos novos spawnam).
- Convencao de display ao usuario: `Level = 7 - row_index`.

### HSV — blocos e HP

```python
BLOCK_HSV_LOW  = np.array([0, 60, 60])     # captura tambem blocos escuros (roxo)
BLOCK_HSV_HIGH = np.array([179, 255, 255])
HP_TEXT_HSV_LOW  = np.array([0, 0, 0])     # texto escuro sobre fundo colorido
HP_TEXT_HSV_HIGH = np.array([179, 255, 99])
MIN_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 0.15)
MAX_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 4.5)
```

`BLOCK_COLOR_RANGES` (config.py:79) lista (nome, low, high) por cor de bloco
(`roxo, vermelho, laranja, magenta, amarelo_ouro, laranja_escuro, rosa, verde,
laranja2, azul, violeta, ciano`). Usada para:
- Isolar o texto do HP dentro do hull colorido (`_get_block_text_mask`).
- Validar se um contorno 2x2 e um bloco grande real (uma cor > 40% da ROI).
- Gate adicional `_qualifies_as_big_block` (dominancia >= 80% da area
  mascarada).

### HSV — power-ups

```python
POWERUP_LASER_HSV_LOW/HIGH       = azul  #008beb   (H 97..107)
POWERUP_EXTRA_BALL_HSV_LOW/HIGH  = ciano #00edd3   (H 81..91)
POWERUP_B_HSV_LOW/HIGH           = ouro  #ffd539   (H 20..26)
```

### Bola

```python
BALL_RADIUS   = max(3, int(BLOCK_SIZE * 0.50))  # 32 px (ver notes.md item 2)
BALL_SPEED    = 8.0
SIM_STEP_SIZE = 2.0
MAX_SIM_STEPS = 2000
```

### Deteccao de fim de rodada (4 regioes absolutas com cor rosa)

```python
ROUND_END_REGIONS = [
    (1445, 920, 22, 29),
    (1544, 919, 25, 30),
    (1649, 922, 22, 28),
    (1748, 919, 27, 32),
]
ROUND_END_HSV_LOW  = np.array([163, 150, 180])   # rosa #e53572
ROUND_END_HSV_HIGH = np.array([177, 255, 255])
```

Threshold: cada regiao "hita" se >10% dos pixels caem na faixa. Rodada acabou
quando >= 2 das 4 hitam.

### Scoring por fileira

```python
ROW_MULTIPLIERS = {
    0: 100.0,  # GAME OVER IMINENTE (L7)
    1: 50.0,   # PERIGO (L6)
    2: 2.5,    3: 1.5,
    4: 1.0,    5: 0.8,    6: 0.5,   # L1 (topo)
}
SCORE_HIT_BLOCK            = 1
SCORE_DESTROY_BLOCK        = 50
SCORE_HIT_POWERUP          = 200
SCORE_MULTI_BOUNCE         = 5
SCORE_LOOP_BONUS           = 100
SCORE_ROW0_SURVIVE_PENALTY = 100
DESTROY_WEIGHT             = 5    # bonus por bloco destruido x ROW_MULTIPLIER
EXTRA_BALL_WEIGHT          = 30   # bonus por bola extra coletada
```

Ranking em `find_best_angle`:
- Se existe bloco em row 0 ou 1: ordena por `(danger_hits, weighted_hits, total_hits)` desc.
- Caso contrario: `(weighted_hits, total_hits)` desc.

### Angulos e execucao

```python
ANGLE_MIN  = 10
ANGLE_MAX  = 170
ANGLE_STEP = 1
BG_VALUE_THRESHOLD  = 80
BALL_COUNT_REGION   = (1308, 824, 536, 33)
BALL_COUNT_HSV_LOW  = np.array([0, 0, 140])
BALL_COUNT_HSV_HIGH = np.array([179, 50, 255])
SPEED_UP_X, SPEED_UP_Y = 1789, 958
SPEED_UP_DELAY = 0.3
TURN_WAIT_TIME = 3.5
PRESCAN_DELAY  = 6.0
DRAG_DISTANCE  = 150
DRAG_DURATION  = 0.15
```

---

## vision.py — deteccao

### Estruturas

- `Block` (dataclass): `x, y, w, h, cx, cy, block_type, hp, vertices,
  tri_orientation, row_index, col_index, hp_mask`. `hp_mask` guarda a
  `text_mask` gerada por `_get_block_text_mask` — alimenta a janela
  `DEBUG_MASK_HP`.
- `PowerUp`: `cx, cy, radius, power_up_type`.
  `PowerUpType` = `EXTRA_BALL | LASER_H | LASER_V | BONUS_B | UNKNOWN`.
- `BlockType` = `SQUARE | TRIANGLE | UNKNOWN`;
  `TriangleOrientation` = 4 cantos + `UNKNOWN`.
- `GameState`: blocos, power-ups, bola, `detected_ball_count`, `game_over`,
  `white_ratio`, `raw_frame`, `debug_frame`.

### Debug frame (cores)

- Bloco 1x1 normal: azul `(255, 0, 0)`.
- Triangulo: vermelho `(0, 0, 255)`.
- Bloco grande (w > 1.5 BLOCK_SIZE): verde `(0, 255, 0)`.
- Power-up: circulo ciano `(255, 255, 0)`.
- Bola: circulo vermelho.
- Label por bloco: `f"{hp} L{display_level}C{col_index}"` no centro.

### Leitura do numero de bolas — `read_ball_count_template()`

Template matching em `BALL_COUNT_REGION` contra `templates/digits/` (28px, gerado
com `calibrate_digits.py`). Filtra pelo HSV claro (`BALL_COUNT_HSV_LOW/HIGH`),
pula o primeiro componente conectado ("x") e concatena os digitos restantes.
Fallback: `count_balls(frame)` — contagem de blobs brancos com area proxima a
`pi * BALL_RADIUS^2` na base inferior 20% do frame.

### `_read_round_number()` — ORFA

Ainda esta no codigo (template matching em `1527,191,101x50` filtrando branco
`S<=30, V>=220`), mas NAO e chamada pelo pipeline atual. Sobrou de uma versao
em que `round_number` alimentava o HP; hoje o HP e OCR puro contra
`templates/hp_digits/`. Candidato a remocao.

### Pipeline de blocos — `detect_blocks(frame)`

1. HSV + `cv2.inRange(BLOCK_HSV_LOW..HIGH)` -> `raw_mask`.
2. Morfologia: `CLOSE 3x3 x2` + `OPEN 3x3 x1` -> `mask`.
3. `findContours(RETR_EXTERNAL)` na `mask`.
4. Para cada contorno:
   - `area < MIN_BLOCK_AREA` ou `w<35 / h<35`: descarta (provavelmente power-up).
   - `area > MAX_BLOCK_AREA`: **big-area split** em 3 fases (abaixo).
   - Caso normal: mapeia na grid via `_get_col_index`/`_get_row_index`, com
     fallbacks de overshoot (< BLOCK_SIZE/2) e centroide (`cv2.moments`).

### `_qualifies_as_big_block(hsv, x, y, w, h)` — GATE 3-checks (NOVO)

Introduzido no commit `87f2f33`. Filtro comum usado tanto no caminho normal 2x2
quanto no `big-area phase 2`, garantindo que a categoria "bloco 2x2 unico"
so seja emitida quando faz sentido:

1. **Dimensao**: `w` e `h` precisam estar em `2*BLOCK_SIZE ± 15%`
   (~108..147 px). Descarta clusters 1x2, 3x3, esticados.
2. **Cor uniforme dominante**: passa todas as `BLOCK_COLOR_RANGES` no ROI,
   soma pixels ativados; a cor top precisa cobrir >= 80% do total mascarado.
   Duas cores dividindo o ROI (ex: roxo 55% + vermelho 40%) => rejeita.
3. **Contagem de HPs**: aplica `_get_block_text_mask` para isolar o texto
   escuro, faz `connectedComponentsWithStats`, agrupa em "clusters" por
   proximidade (`BLOCK_SIZE/3` em X, `BLOCK_SIZE/4` em Y). Um 2x2 legitimo
   tem UM cluster (o HP central). 2+ clusters => 2+ blocos distintos, rejeita.

Rejeicoes sao printadas com `[big-block reject ...]` para auditoria.

### Big-area split (3 fases)

Quando um contorno excede `MAX_BLOCK_AREA` (fusao morfologica de varios blocos):

1. **Fase 1 — survivors**: para cada celula da grid coberta pelo bounding rect,
   exige `fill >= 0.40` na mask. Cada sobrevivente ja e classificado
   individualmente como SQUARE ou TRIANGLE via `raw_mask` (hull_fill < 0.60
   => triangulo). Log `[big-area phase1] survivors=...`.
2. **Fase 2 — agrupar 2x2 por cor uniforme + gate**: itera pares de linhas/colunas
   adjacentes. Emite UM bloco 2x2 quando:
   - as 4 celulas estao em `survivors`,
   - alguma `BLOCK_COLOR_RANGES` cobre > 40% da ROI 2x2,
   - `_qualifies_as_big_block` passa (dim + dominancia + hp-count).
   HP e lido com `is_2x2=True` (pad minimo `pad=2`, `min_area=50`). As 4
   celulas viram "consumed". Log `[big-area 2x2]`.
3. **Fase 3 — split em 1x1**: celulas nao consumidas viram blocos individuais
   (`is_tri` propagado quando triangulo). Log `[big-area split]`.

### Caminho normal — span 1x1, 2x2 candidato, fundido

- **1x1**: classifica via `raw_mask` sem morfologia (`hull_fill < 0.60` =>
  triangulo). Se `_get_triangle_orientation` retorna `UNKNOWN`, degrada para
  SQUARE. `_hp` com `is_tri` propagado.
- **2x2 candidato** (`cols_span==2 and rows_span==2`): passa 2 gates encadeados:
  - Gate de cor uniforme (uma `BLOCK_COLOR_RANGES` cobre > 40%).
  - `_qualifies_as_big_block(hsv, x, y, w, h)`.
  Se passa, emite 1 bloco 2x2. Se falha, split em 1x1 individuais com
  `fill > 15%` por celula. O gate antigo de cell-fill (>50% por celula) foi
  DESABILITADO porque HP de 3 digitos deixa buracos que derrubavam o fill
  (ex: rosa "107" no bloco verde 2x2).
- **Fundido (2x1, 1x2, etc.)**: split por celula com `fill > 15%`,
  classificacao individual via `raw_mask` (inclusive orientacao de triangulo).

### Atribuicao de HP — `_read_hp(...)`

HP e OCR puro contra `templates/hp_digits/<digit>.png` (20px), sem fallback
para `round_number` e sem tracking entre frames.

- `_get_block_text_mask(roi_bgr, is_triangle)`:
  - Escolhe a `BLOCK_COLOR_RANGES` de maior cobertura no ROI e usa o convex
    hull daquela mask colorida como `block_region` (para quadrados). Para
    triangulos, `block_region = ROI inteira` — o hull cortava o digito que
    encosta na hipotenusa.
  - `dark = cv2.inRange(v_channel, 0, 130)`  (antigo: `0..90`).
  - `text_mask = dark & block_region`.
  - Limpa ruido com `MORPH_OPEN 1x1` (antigo `2x2`).
  - Se `is_triangle=True`, imprime log `[TRI hp_mask] ...`.
- Padding da ROI:
  - Triangulo: `pad_x = pad_y = 2` (digito deslocado pro centroide colorido).
  - 2x2 consolidado: `pad_x = pad_y = 2` (HP de 3 chars ocupa quase toda a
    largura — pad maior cortava as pontas).
  - Bloco 1x1: `pad = max(2, w // 6)` (padrao).
- `min_area` para componentes:
  - 2x2: fixo em 50 (proporcional descartava digitos das pontas em "107").
  - Outros: `(w * h) // 100`.
- Se `text_mask` tem <5 pixels ou nenhum componente valido: retorna `hp=1`.
- Templates de HP: `_load_hp_digit_templates()` le `templates/hp_digits/<digit>.png`
  (nomes de 1 caractere apenas), normaliza para altura 20px, binariza. Ignora
  as amostras brutas `<digito>_<timestamp>.png`.
- Score por template = fracao de pixels iguais. Threshold minimo: 0.6.
- Retorna `(hp_int, text_mask)`. `text_mask` alimenta `Block.hp_mask` e a
  janela `DEBUG_MASK_HP`.

### Orientacao de triangulos — `_get_triangle_orientation(raw_roi)`

Centroide (`cv2.moments`) do mask na celula vs centro da ROI. `dx/dy` decidem
o quadrante (`TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT`). Se
`|dx|` e `|dy|` ambos < 5 => `UNKNOWN`, degrada para SQUARE.

### Mapeamento na grid

```python
_MARGIN = 3
c_left  = _get_col_index(x + _MARGIN)
c_right = _get_col_index(x + w - _MARGIN)
r_top   = _get_row_index(y + _MARGIN)
r_bot   = _get_row_index(y + h - _MARGIN)
```

- **Overshoot** < `BLOCK_SIZE//2`: se um lado vira -1 mas o oposto e valido,
  assume o mesmo indice (gating contra cair fora da grid).
- **Centroide**: ultimo recurso via `cv2.moments(cnt)`.
- Se qualquer um continuar -1, o contorno e descartado.

### Power-ups — `detect_powerups(frame)`

HSV + circularidade > 0.6 + raio 7..15 px. Tipos:
- `EXTRA_BALL` (+ ciano) — soma bola.
- `LASER_H / LASER_V` (azul <>): orientacao decidida por variancia das medias
  por linha vs coluna (`h_proj > v_proj` => `LASER_V`).
- `BONUS_B` (B amarelo ouro).

### Game over — `detect_game_over(frame)`

Pixels > 200 no canal gray > 25% do frame. Ao disparar, `scan_game` retorna um
`GameState(blocks=[], powerups=[], ball=None, detected_ball_count=0,
game_over=True, white_ratio=...)`. Bug antigo de TypeError (falta de
`detected_ball_count`) ja corrigido.

### Fim de rodada — `detect_round_end()`

Captura as 4 `ROUND_END_REGIONS` e testa cobertura da faixa rosa > 10%. Retorna
True se >= 2 das 4 hitam.

---

## physics.py — simulacao

### `SimResult`

```python
angle: float
score: float               # display apenas
total_hits: int
weighted_hits: float       # hits x ROW_MULTIPLIERS (ranking primario)
danger_hits: int           # hits efetivos em rows 0-1 (ranking primario c/ perigo)
blocks_destroyed: int
powerups_collected: int
extra_balls_collected: int
path: List[Tuple]          # so no re-simulate do melhor angulo
collisions: List[Tuple]    # posicoes das colisoes (uso: animate_dryrun)
hit_details: List[str]
```

### `_sim_one_ball` — simulacao de UMA bola

Retorna `(hit_sequence, new_pu_indices, path, collisions)`. Movimento por
`SIM_STEP_SIZE / BALL_SPEED` por step (limite `MAX_SIM_STEPS`). Encerra
quando a bola "subiu" (>= `start_y - 2r`) e depois voltou para
`effective_bottom = max(WALL_BOTTOM, start_y + radius + 2)`.

**Colisoes:**
- **AABB** (blocos quadrados): face mais penetrada decide a normal; fallback
  pela direcao centro-bola.
- **Triangulos**: AABB como fast-reject + verificacao real de proximidade do
  poligono (`_point_near_polygon`). Distingue entrada pelo cateto vs
  hipotenusa via dot product com a normal diagonal (TL `(+, +)`, TR `(-, +)`,
  BL `(+, -)`, BR `(-, -)` normalizadas). Orientacao `UNKNOWN` => puro AABB.
- **Power-ups** (circular): coletados pelo set `new_pus`, efeito aplicado
  pos-bola em `simulate_trajectory`.

`record_path=True` amostra `path` a cada 4 steps (8 px por indice) e registra
`collisions` em paredes, blocos, triangulos.

### `simulate_trajectory` — loop de bolas com batch

1. **Power-ups coletados:**
   - `LASER_H/LASER_V`: `_apply_laser` aplica 1 de dano por bola do batch
     corrente em todos os blocos da mesma linha/coluna (com `abs(delta) <
     BLOCK_SIZE/2`); forca `batch = 1` para evitar re-simulacao com estado
     alterado.
   - `EXTRA_BALL`: `+SCORE_HIT_POWERUP`, `+EXTRA_BALL_WEIGHT` no weighted.
   - Outros: `+SCORE_HIT_POWERUP`.
2. **Break** se `hit_seq` vazio.
3. **Batch**: bolas que seguem identica trajetoria sao agrupadas ate destruir
   o proximo bloco (calculado por `ceil(hp / nh)` por bloco). Forcado a 1 se
   laser disparou.

Apos o loop: penalidade `-SCORE_ROW0_SURVIVE_PENALTY * hp` por bloco da
row 0 que sobreviveu; bonus `SCORE_LOOP_BONUS * (total_hits // 10)` se
`total_hits > 10`.

### `find_best_angle`

Simula 1 em 1 grau (`ANGLE_MIN..ANGLE_MAX`). Ordena conforme presenca de
blocos em row 0-1. Re-simula o melhor angulo com `record_path=True` para
alimentar overlay/animate (overhead aceito, uma passada extra).

---

## main.py — loop principal

### Flags

```
python main.py                  # bot normal
python main.py --debug          # janela de debug com trajetoria
python main.py --dry-run        # visualizacao continua sem disparos
python main.py --animate        # dispatcha tools/animate_dryrun.run_animate()
python main.py --balls N        # forca N bolas (ignora deteccao visual)
python main.py --delay N        # delay inicial em segundos (default 1.0)
python main.py --max-turns N    # encerra apos N turnos
python main.py --smart-wait     # deteccao visual para esperar turno (nao usado
                                #   pelo loop atual, que usa detect_round_end)
```

### Fluxo por turno

1. `time.sleep(3)`
2. `scan_game()` -> `update_level_display(...)` (thread de debug consome).
3. `run_turn(state)`: cheka `game_over`, monta print por level, chama
   `find_best_angle`, imprime `melhor angulo / hits / score / dano por hit`.
4. `execute_shot(...)` (swipe + click no botao de acelerar).
5. Poll `detect_round_end()` a cada 0.3s (timeout 30s).
6. `time.sleep(2.0)` (blocos descem e estabilizam).
7. Volta para 1.

### Deteccao de jogo travado

`fingerprint = frozenset((row_index, round(cx / BLOCK_SIZE)) for b in blocks)`.
Se identico por 20 turnos consecutivos, clica `(1580, 961)` para destravar.

### Game over

Loga em `historico.txt` (`YYYY-MM-DD HH:MM:SS | Turnos: N | Bolas: M`),
executa 3 sequencias de clique em `(1576, 781)` e `(1584, 838)` com sleep de
4s entre elas, reseta estado interno (`turn`, `ball_count`, fingerprint,
stale_turns) e recomeca.

### Thread de debug — `tools/visualizar_masks.py:run_masks_window`

Daemon thread. Mostra:
- **"Blocos + PowerUps"**: OR de todas as masks de cor (COMBINADA).
- **"Qtd Bolas"**: strip de `BALL_COUNT_REGION` + numero lido por template.
- **"Fim de Rodada"**: 4 mini-strips das `ROUND_END_REGIONS` + status
  "RODADA ACABOU" / "Aguardando... (N/4)".
- **"Blocos por Level"**: `debug_frame` do ultimo scan com grade e labels
  de coluna (C0..C6) e level (opcional, comentado).
- **"DEBUG_MASK_HP"**: mosaico 7x7 (rows invertidas: L1 em cima) onde cada
  tile 70px mostra a `text_mask` do bloco + HP lido (verde no canto).

Sincronizado com o main via `update_level_display(debug_frame, blocks,
ball_count)` (variaveis globais).

---

## executor.py — disparo

- `angle_to_screen_coords`: converte `(ball_x, ball_y, angle_deg)` +
  `DRAG_DISTANCE=150` em `(sx, sy, tx, ty)` absolutos na tela (soma
  `GAME_REGION[:2]`).
- **Metodo padrao** (`_base_shot`): `execute_shot_swipe` — `moveTo` + `drag`.
- Alternativas prontas: `execute_shot_tap` (click simples no alvo),
  `execute_shot_hold_release` (`mouseDown` + `moveTo` + `mouseUp`).
- `speed_up_click()` apos disparo: `time.sleep(SPEED_UP_DELAY)` + clique em
  `(SPEED_UP_X, SPEED_UP_Y)`. `pyautogui.FAILSAFE=True` (canto sup esq aborta).
- `wait_for_turn_end_smart`: monitora a bola voltando perto de `WALL_BOTTOM`;
  considera estavel quando esta perto por 3 checks consecutivos. **Nao e usado
  pelo main loop atual** (que usa `detect_round_end`).

---

## main_vision.py — loop de validacao de visao

Roda `scan_game()` continuamente, imprime no console e mantem a thread
`run_masks_window` viva. Sem fisica, sem simulacao, sem mouse. Util para
validar deteccao de blocos, HP e power-ups.

```
python main_vision.py
python main_vision.py --delay 3 --interval 0.5
```

---

## monitor.py — monitor lado-a-lado

Janela unica com painel esquerdo (deteccao) + direito (simulacao top-5 em
cinza + melhor em branco + seta). Sem execucao no mouse. Escala 0.5x, FPS
configuravel.

```
python monitor.py
python monitor.py --balls 5 --fps 10
python monitor.py --no-sim
```

---

## tools/animate_dryrun.py — dry-run animado

Duas fases em loop, respondendo a ESC/`q`:
- **Phase 1** (5 s): leque com TODOS os `ANGLE_MIN..MAX` a partir da bola —
  cinza translucido, top-5 amarelo, melhor branco grosso.
- **Phase 2**: bola percorrendo o `best.path` em ~60 FPS, com trail crescente,
  marcadores vermelhos nas colisoes ja visitadas e ate `ball_count` bolas
  espacadas por `BALL_SPACING=3` indices (~24 px). No fim, freeze 3 s e volta
  para Phase 1.

Invocado por `python main.py --animate` (delay inicial + `run_animate()`).

---

## Templates de HP — pipeline de captura

### Por que existe

A leitura do HP e pure template matching contra `templates/hp_digits/<digit>.png`
(0..9, ~20 px). Para que o OCR funcione bem, esses 10 templates sao construidos
a partir de multiplas amostras reais do jogo (uma media por digito).

### Captura — `capture_hp_templates.py`

Loop interativo: roda `scan_game()`, e para cada bloco mostra cada componente
conectado da `hp_mask` (a mesma filtragem usada por `_read_hp`) ampliado 10x
numa janela OpenCV. Teclas (foco NA janela "Template Capture"):

- `0-9`: salva o crop como `templates/hp_digits/<digito>_<timestamp>.png`.
- `s`: skip esse componente.
- `n`: pula resto do bloco.
- `f`: pula resto do frame (re-captura).
- `q`: sai.

Reaproveita `_components()` (replica da logica de `_read_hp`) para mostrar
exatamente os crops que o pipeline vai ler. Deduplica via
`seen = set((shape, bytes))`.

### Consolidacao — `consolidate_hp_templates.py`

Para cada digito 0..9:

1. Le todas as amostras `<digito>_*.png` na pasta.
2. Redimensiona para altura 20 px mantendo aspect ratio.
3. Escolhe largura mediana como `target_w`, renormaliza tudo.
4. Media dos pixels + `threshold 127`.
5. Salva como `templates/hp_digits/<digito>.png`.

`_load_hp_digit_templates` em `vision.py` filtra por nome de 1 caractere,
entao manter as amostras `<digito>_<ts>.png` na pasta nao atrapalha runtime.

Pasta atual: ~150 arquivos (10 templates consolidados + ~140 amostras
brutas, predominantemente para 1, 2 e 0).

---

## Dependencias

```
opencv-python
numpy
mss
pyautogui
```

---

## Historico de decisoes relevantes

### Recalibracao de grid e paredes (commit 87f2f33)

- `GAME_REGION = (1352, 244, 452, 578)` (era `(1353, 245, 451, 577)`).
- `WALL_LEFT = 0 / WALL_RIGHT = 449 / WALL_TOP = 2 / WALL_BOTTOM = 575` —
  paredes coincidem quase exatamente com a borda do playfield agora.
- `LEVEL_BOTS[0] = 513` — antes `WALL_BOTTOM = 575` fazia o codigo detectar o
  player (bola parada na base) como bloco.

### `_qualifies_as_big_block` (3-check gate)

Adicionado no mesmo commit. Aplica dimensao 2*BLOCK_SIZE ±15%, dominancia de
cor >= 80%, e contagem de HP-clusters (deve ser 1). Chamado tanto no caminho
normal 2x2 quanto na `big-area phase 2`. Fecha os falsos positivos onde 4
blocos coloridos diferentes iam sendo agrupados como "2x2 unico".

### HP via OCR puro (sem round_number, sem registry)

Cada frame e detectado do zero. Nao ha tracking entre frames ou rodadas.
`_read_hp` faz template matching contra `templates/hp_digits/` (20 px) e
retorna 1 se falhar. Sem fallback para `round_number`. Templates construidos
via `capture_hp_templates.py` (amostras reais) + `consolidate_hp_templates.py`
(media + binarizacao). A funcao `_read_round_number()` ainda esta no codigo
mas nao e chamada — candidato a remocao.

### Big-area split em 3 fases

Quando o contorno excede `MAX_BLOCK_AREA`, coleta cells sobreviventes por
fill >= 40%, tenta agrupar 2x2 reais por dominancia de cor (>40%) + gate
`_qualifies_as_big_block`, e o que sobrar vira 1x1 individual. Emissao de
2x2 unico e essencial para HP de 3 digitos (ex: verde 2x2 com "107") nao
virar 4 cells lendo pedacos.

### Distincao 2x2 real vs fusao

Por uniformidade de cor + gate. O gate antigo por cell-fill (>50% por celula)
foi desabilitado (HP grande deixava buracos que derrubavam o fill).

### Triangulos

- Classificados por `hull_fill < 0.60` na `raw_mask` (sem morfologia que
  infla contornos).
- Orientacao via centroide do mask. Se ambiguo (`|dx|, |dy| < 5`), degrada
  para SQUARE.
- Colisao distingue cateto (rebate como parede) vs hipotenusa (reflexao pela
  normal diagonal) via dot product centro-bola vs normal.
- OCR de HP em triangulo usa ROI inteira como `block_region` (sem hull),
  para nao cortar o digito que encosta na hipotenusa. Padding minimo
  (`pad=2`) porque o digito fica deslocado pro centroide colorido.
- Threshold de "dark" para texto: `V < 130` (era `V < 90`) — recupera
  digitos que ficam com tons medios sobre fundo escuro (roxo, magenta).

### Power-ups na simulacao

- Coletados na colisao com a bola (`new_pus`); efeito aplicado pos-bola.
- LASER aplica `_apply_laser` instantaneo (1 dano por bola do batch) e
  forca `batch = 1` para evitar re-simulacao com estado alterado.
- EXTRA_BALL adiciona `EXTRA_BALL_WEIGHT` no weighted (peso entre row 0/1
  e o resto).

### Ranking de jogada

Com blocos em row 0-1: `(danger_hits, weighted_hits, total_hits)` desc.
Sem perigo: `(weighted_hits, total_hits)` desc.

### Mapeamento na grid robusto

Fallbacks de overshoot (lado fora da grid mas oposto valido) e centroide
(`cv2.moments`) para nao perder blocos perto da borda.

### Batch optimization na simulacao

Agrupa bolas que seguem a mesma trajetoria ate destruir um bloco, evitando
re-simular cada uma. Forcado a 1 quando laser e ativado.

### `BLOCK_HSV_LOW` reduzido para `S=60, V=60`

Captura tambem blocos escuros como o roxo.

### Sem tracking entre frames

Cada frame e detectado do zero. Removido qualquer registry persistente de HP.
Erros de OCR nao se propagam, mas o bot perde info quando um bloco "pisca"
entre deteccoes.

---

## Pontos conhecidos em aberto (`notes.md`)

- **BALL_RADIUS = 32 px** (`BLOCK_SIZE * 0.50`): pode estar inflando colisoes.
  Valor real do jogo seria ~0.10-0.15 do bloco. Ainda em avaliacao — linha
  alternativa `int(BLOCK_SIZE * 0.22)` fica comentada no `config.py`.
- **count_balls usa `BALL_RADIUS` pra estimar area**: `pi * 32^2 = 3217 px^2`.
  Se a bola real for menor, o fallback sempre retorna 1.
- **`detect_blocks` capta power-ups como blocos** pelo HSV amplo
  (`[0,60,60]-[179,255,255]`). Mitigado pelo filtro `w<35 or h<35`, mas
  ainda nao ha dedup explicita contra `detect_powerups`.
- **`_read_round_number` orfa**: ainda existe no codigo mas nao e chamada.
  Candidato a remocao se nao voltar a usar.
- **`historico.txt` sempre grava `Bolas: 1`**: contagem de bolas gravada no
  log e o valor local `ball_count`, que nao e atualizado pela deteccao real
  entre turnos (so via `--balls` ou pela primeira leitura). Nao afeta a
  jogabilidade porque `run_turn` usa `state.detected_ball_count`.

---

## Recordes recentes (historico.txt)

- 2026-06-23 23:39:54 — **63 turnos** | RECORDE
- 2026-06-24 00:01:17 — 51 turnos
- 2026-06-17 01:28:06 — 48 turnos
- 2026-06-17 01:19:52 — 45 turnos
- 2026-07-01 22:52:06 — 42 turnos  (primeira partida pos-recalibracao)

Media das partidas de 2026-07-01 (4 partidas): ~29 turnos. Media das ~20
partidas mais recentes: ~30 turnos por run, com `Bolas: 1` fixo no log
(ver pontos em aberto).
