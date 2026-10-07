"""
config.py — Constantes globais do BBTAN Bot.

Ajuste GAME_REGION para bater com a janela do seu emulador.
Use o script calibrate.py (incluso) para descobrir os valores certos.
"""

import numpy as np

# ─────────────────────────────────────────────
# REGIÃO DO JOGO NA TELA (x, y, largura, altura)
# Rode calibrate.py para pegar esses valores
# ─────────────────────────────────────────────
# antigo: (1353, 245, 451, 577)
GAME_REGION = (1352, 244, 452, 578)

# ─────────────────────────────────────────────
# ESCALA — todos os valores em pixel deste arquivo foram medidos com o jogo em
# REF_LARGURA px de largura (perfil de referência). O BBTAN escala por igual, então
# se o jogo tiver outro tamanho basta multiplicar: comprimentos por ESCALA
# (px) e áreas por ESCALA² (area_px). Com ESCALA = 1 tudo fica como medido.
# ─────────────────────────────────────────────
REF_LARGURA = 452
ESCALA = GAME_REGION[2] / REF_LARGURA


def px(valor: float) -> int:
    """Comprimento medido no perfil de referência -> px no tamanho atual do jogo."""
    return int(round(valor * ESCALA))


def area_px(valor: float) -> float:
    """Área (px²) medida no perfil de referência -> px² no tamanho atual do jogo."""
    return valor * ESCALA * ESCALA


def regiao_hud(dx: float, dy: float, w: float, h: float) -> tuple:
    """
    Região do HUD (fora do tabuleiro) -> (x, y, w, h) absoluto na tela.
    dx/dy são medidos a partir do canto superior esquerdo do GAME_REGION, no
    perfil de referência — assim o HUD acompanha o jogo quando a janela muda
    de lugar ou de tamanho.
    """
    return (GAME_REGION[0] + px(dx), GAME_REGION[1] + px(dy), px(w), px(h))


def ponto_hud(dx: float, dy: float) -> tuple:
    """Ponto do HUD relativo ao canto do GAME_REGION -> (x, y) absoluto na tela."""
    return (GAME_REGION[0] + px(dx), GAME_REGION[1] + px(dy))


# Margens internas da área jogável (pixels relativos à GAME_REGION)
# antigo: WALL_LEFT = 5
WALL_LEFT = px(0)
# antigo: WALL_RIGHT = GAME_REGION[2] - 5  (=446)
# antigo: WALL_RIGHT = 449
WALL_RIGHT = px(449)
# antigo: WALL_TOP = 5
# antigo: WALL_TOP = 2
WALL_TOP = px(2)
# antigo: WALL_BOTTOM = GAME_REGION[3] - 10  (=567)  # acima do robô/bola
# antigo: WALL_BOTTOM = 575
WALL_BOTTOM = px(575)

# Grid do BBTAN: 7 colunas
GRID_COLUMNS = 7
BLOCK_SIZE = GAME_REGION[2] // GRID_COLUMNS  # pixels por bloco

# Limites calibrados de cada coluna (via calibrate_columns.py)
# antigo: COL_LEFTS = [4, 67, 132, 195, 258, 321, 386]
# antigo: COL_RIGHTS = [63, 128, 191, 255, 318, 382, 444]
COL_LEFTS = [px(x) for x in (4, 67, 132, 195, 258, 321, 386)]
COL_RIGHTS = [px(x) for x in (63, 128, 191, 255, 318, 382, 444)]

# Limites de cada fileira, relativos ao frame capturado (0 = mais baixa/perigosa)
# Computados a partir de _GRID_TOP_Y e BLOCK_SIZE para cobertura contígua (sem gaps)
# antigo: _GRID_TOP_Y = 67
_GRID_TOP_Y = px(67)  # Y do topo da fileira mais alta
LEVEL_TOPS = [_GRID_TOP_Y + (6 - i) * BLOCK_SIZE for i in range(7)]
LEVEL_BOTS = [_GRID_TOP_Y + (7 - i) * BLOCK_SIZE - 1 for i in range(7)]
# antigo: LEVEL_BOTS[0] = 514 (=_GRID_TOP_Y + 7*BLOCK_SIZE - 1). Cortava 1px
# do bloco da fileira mais baixa; ajustado pra bater exato com a base do bloco.
# antigo (tentativa anterior): LEVEL_BOTS[0] = WALL_BOTTOM = 575, mas esticava
# a fileira ate a base do playfield e fazia o codigo detectar o player como bloco.
# antigo: LEVEL_BOTS[0] = 513
LEVEL_BOTS[0] = px(513)
# Centro Y de cada fileira. Derivado — consumido apenas por tools/ (calibrate_levels,
# visualizar_masks). O bot em si usa LEVEL_TOPS/LEVEL_BOTS via vision._get_row_index.
LEVELS = [(t + b) // 2 for t, b in zip(LEVEL_TOPS, LEVEL_BOTS)]


# ─────────────────────────────────────────────
# CORES HSV — extraídas da screenshot
# (OpenCV usa H: 0-179, S: 0-255, V: 0-255)
# ─────────────────────────────────────────────

# --- POWER-UPS ---

# Power-Up Laser Horiz/Vert (<>) — Azul (#008beb)
POWERUP_LASER_HSV_LOW = np.array([97, 180, 180])
POWERUP_LASER_HSV_HIGH = np.array([107, 255, 255])

# Power-Up Bola Extra (+) — Verde Ciano (#00edd3)
POWERUP_EXTRA_BALL_HSV_LOW = np.array([81, 180, 180])
POWERUP_EXTRA_BALL_HSV_HIGH = np.array([91, 255, 255])

# Power-Up Moeda/Bônus B (B) — Amarelo Ouro (#ffd539)
POWERUP_B_HSV_LOW = np.array([20, 180, 200])
POWERUP_B_HSV_HIGH = np.array([26, 255, 255])

# Power-Up Direção Aleatória (3 setas pra cima) — Roxo
# Medido no log bot_20260929_173907 (turno 2): H 134-144, S ~215, V 255.
# Sobrepõe o roxo dos blocos, mas o filtro de raio de detect_powerups
# (MIN/MAX_POWERUP_RADIUS) descarta qualquer bloco.
POWERUP_RANDOM_HSV_LOW = np.array([131, 150, 150])
POWERUP_RANDOM_HSV_HIGH = np.array([145, 255, 255])


# --- BLOCOS POR COR (usado para isolar texto de HP via inversão) ---

BLOCK_COLOR_RANGES = [
    # (nome, low, high)
    ("roxo", np.array([126, 200, 150]), np.array([143, 255, 255])),
    ("vermelho", np.array([0, 200, 150]), np.array([9, 255, 255])),
    ("laranja", np.array([16, 200, 150]), np.array([22, 255, 255])),
    ("laranja_escuro", np.array([9, 200, 150]), np.array([16, 255, 255])),
    ("magenta", np.array([144, 180, 120]), np.array([162, 255, 255])),
    ("amarelo_ouro", np.array([22, 200, 150]), np.array([28, 255, 255])),
    ("rosa", np.array([162, 200, 150]), np.array([179, 255, 255])),
    ("verde", np.array([65, 180, 120]), np.array([78, 255, 255])),
    ("azul", np.array([108, 100, 150]), np.array([120, 255, 255])),  # #5c6ebf
    ("violeta", np.array([118, 110, 160]), np.array([128, 255, 255])),  # #6557cf
    ("ciano", np.array([82, 120, 130]), np.array([92, 255, 255])),  # #42ab9f
]

# BLOCOS — qualquer coisa colorida (exclui fundo escuro V<50)
# BLOCK_HSV_LOW = np.array([0, 80, 100])  # antigo: perdia blocos escuros (roxo)
BLOCK_HSV_LOW = np.array([0, 60, 60])
BLOCK_HSV_HIGH = np.array([179, 255, 255])

# Número de HP — texto escuro sobre fundo colorido.
# O componente V (índice 2) é a fonte única do limiar de "pixel escuro = texto",
# usado nos DOIS caminhos de vision._get_block_text_mask (dentro do hull colorido
# e no fallback por HSV). Antes o caminho principal tinha `inRange(v, 0, 130)`
# hardcoded enquanto o fallback lia V<=99 daqui — dois valores para a mesma decisão.
# antigo (branco): [0, 0, 252] / [175, 255, 255]
# antigo (V<=99): np.array([179, 255, 99]) — divergia do 130 usado na prática
HP_TEXT_HSV_LOW = np.array([0, 0, 0])
HP_TEXT_HSV_HIGH = np.array([179, 255, 130])

# BOLA branca
BALL_HSV_LOW = np.array([0, 0, 200])
BALL_HSV_HIGH = np.array([179, 40, 255])

# Contagem de bolas (OCR) — região na tela onde aparece "x14", "x1", etc.
# As regiões/pontos do HUD abaixo são relativos ao canto do GAME_REGION (regiao_hud).
# antigo: (1324, 829, 483, 36)
# antigo (absoluto): BALL_COUNT_REGION = (1308, 824, 536, 33)
BALL_COUNT_REGION = regiao_hud(-44, 580, 536, 33)
# antigo: [0, 0, 200] / [179, 41, 255]
BALL_COUNT_HSV_LOW = np.array([0, 0, 140])
BALL_COUNT_HSV_HIGH = np.array([179, 50, 255])

# Indicador de fim de rodada — 4 regiões absolutas na tela com cor rosa (#e53572)
# Quando as 4 regiões contêm essa cor, a rodada acabou e o jogo pode ser analisado
# antigo (absoluto): (1445, 920, 22, 29), (1544, 919, 25, 30),
#                    (1649, 922, 22, 28), (1748, 919, 27, 32)
ROUND_END_REGIONS = [
    regiao_hud(93, 676, 22, 29),
    regiao_hud(192, 675, 25, 30),
    regiao_hud(297, 678, 22, 28),
    regiao_hud(396, 675, 27, 32),
]
# #e53572 → RGB(229,53,114) → OpenCV HSV: H≈170, S≈196, V≈229
ROUND_END_HSV_LOW = np.array([163, 150, 180])
ROUND_END_HSV_HIGH = np.array([177, 255, 255])

# Botão "retrieve" (recolher bolas) — retângulo roxo (#7921dc) que fica na tela
# ENQUANTO a jogada está acontecendo. Presente = rodada em andamento; sumiu =
# pode começar a pensar na próxima. Região na tela (x, y, w, h).
# antigo (absoluto): RETRIEVE_BUTTON_REGION = (1485, 947, 185, 38)
RETRIEVE_BUTTON_REGION = regiao_hud(133, 703, 185, 38)
# #7921dc → RGB(121,33,220) → OpenCV HSV: H≈134, S≈217, V≈220
RETRIEVE_HSV_LOW = np.array([129, 170, 170])
RETRIEVE_HSV_HIGH = np.array([139, 255, 255])
# Fração mínima da região que precisa estar roxa pra contar como "botão presente".
# Sem o botão (ícones da loja no lugar) a máscara mede 0%.
RETRIEVE_MIN_FILL = 0.30
# Ponto clicado pra forçar o fim da rodada — dentro do botão.
# Era o clique hardcoded da antiga detecção de "jogo travado" em main.py.
# antigo (absoluto): RETRIEVE_CLICK = (1580, 961)
RETRIEVE_CLICK = ponto_hud(228, 717)

# Tempo máximo (s) de uma rodada. Se o botão retrieve continuar na tela por mais
# que isso, as bolas provavelmente estão num loop quase horizontal — o bot clica
# no botão pra forçar o fim da rodada.
TEMPO_MAX_RODADA = 60 * 3

# ─────────────────────────────────────────────
# DETECÇÃO DE BLOCOS
# ─────────────────────────────────────────────

MIN_BLOCK_AREA = int(
    BLOCK_SIZE * BLOCK_SIZE * 0.15
)  # ~15% da área de 1 bloco (triangulo ~50%)
MAX_BLOCK_AREA = int(
    BLOCK_SIZE * BLOCK_SIZE * 4.5
)  # bloco grande 2x2 (4x) + margem morfológica
POLY_APPROX_FACTOR = 0.04  # fator do approxPolyDP

# Limiares em pixel que antes estavam chumbados em vision.py (valores do perfil
# de referência; comprimentos passam por px, áreas por area_px).
MIN_BLOCK_SIDE = px(35)  # bbox menor que isso é power-up, não bloco
BLOCK_EDGE_MARGIN = px(3)  # recuo da bbox ao mapear bloco -> célula da grade
MIN_POWERUP_AREA = area_px(50)
# antigo: chumbado em vision.detect_powerups como `radius < 7 or radius > 15`
MIN_POWERUP_RADIUS = px(7)
MAX_POWERUP_RADIUS = px(15)
MIN_BALL_AREA = area_px(20)  # bola parada: área do contorno branco
MAX_BALL_AREA = area_px(500)
MIN_HP_DIGIT_AREA_2X2 = area_px(50)  # dígito de HP em bloco 2x2
MIN_BALL_COUNT_DIGIT_AREA = area_px(15)  # dígito do contador "x14"
MIN_BALL_COUNT_DIGIT_H = px(6)
# Folga acima da faixa esperada do contador ao checar se uma leitura fora dela
# é troca conhecida de dígito (vision.escolher_contagem). Bola extra coletada e
# não detectada põe o real acima de [anterior, anterior + "+"].
# antigo: sem folga (0)
CONTADOR_FOLGA = 2


# ─────────────────────────────────────────────
# SIMULAÇÃO DE FÍSICA
# ─────────────────────────────────────────────

# Bola de 4mm in-game. Tela mobile ~65mm de largura, 7 blocos.
# Bloco ~9.3mm, raio da bola = 2mm -> 2/9.3 ~= 0.22 do bloco
# BALL_RADIUS   = max(3, int(BLOCK_SIZE * 0.22))  # pixels

# Raio da bola = 1.77mm. Bloco = 8mm → fator = 1.77/8 ≈ 0.22
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.50))  # antigo: raio 4mm
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.40))
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.22))
# antigo: BALL_RADIUS = 9
BALL_RADIUS = px(9)

# DECORATIVO — não tem efeito nenhum na simulação. physics._sim_one_ball faz
# `vel.normalized() * BALL_SPEED` e depois avança `vel * (SIM_STEP_SIZE / BALL_SPEED)`:
# multiplica e divide pelo mesmo valor, então o avanço é sempre
# `direção × SIM_STEP_SIZE`, independente do que estiver aqui. Quem controla a
# resolução da simulação é o SIM_STEP_SIZE. Mantido só para não quebrar imports.
BALL_SPEED = 8.0

# ── Alcance × precisão da simulação ──
# MAX_SIM_STEPS e SIM_STEP_SIZE são ACOPLADOS: o que governa o alcance da bola é o
# PRODUTO dos dois (quantos px de percurso cabem num voo). Mexer só na precisão
# encurtava o voo em silêncio — ex: SIM_STEP_SIZE 2.0→0.5 com MAX_SIM_STEPS
# 2000→3000 derrubou o orçamento de 4000px para 1500px, e como o playfield tem
# 573px de altura (2×573/sin(19°) = 3520px para subir e voltar), todo ângulo
# abaixo de ~57° era truncado no meio do voo.
# Agora o orçamento é declarado e o nº de steps é derivado dele.
# Os dois escalam juntos: o voo cobre a mesma fração do tabuleiro e o nº de steps
# (custo da simulação) não muda com o tamanho do jogo.
# antigo: SIM_PATH_BUDGET = 4000.0
SIM_PATH_BUDGET = 4000.0 * ESCALA  # px de percurso que a bola pode percorrer por voo
# antigo: SIM_STEP_SIZE = 1.0
SIM_STEP_SIZE = 1.0 * ESCALA  # antigo: 0.5 (2.7x mais lento); antes disso: 2.0
MAX_SIM_STEPS = round(SIM_PATH_BUDGET / SIM_STEP_SIZE)

# ─────────────────────────────────────────────
# SCORING — pesos para a decisão de ângulo
# ─────────────────────────────────────────────
# Todos os pesos abaixo entram num ÚNICO score (physics.compute_score), que é o
# critério de ordenação de find_best_angle. Antes havia duas notas paralelas: os
# SCORE_* alimentavam `score`, que era só impresso, e o ranking usava outra conta.
#
# antigo (decorativos — nunca influenciaram a escolha do ângulo):
#   SCORE_HIT_BLOCK = 1              -> virou W_HIT
#   SCORE_DESTROY_BLOCK = 50         -> removido (redundante com W_DESTROY)
#   SCORE_HIT_POWERUP = 200          -> removido (redundante com W_EXTRA_BALL/W_BONUS_B)
#   SCORE_LOOP_BONUS = 100           -> virou W_LOOP
#   SCORE_ROW0_SURVIVE_PENALTY = 100 -> virou W_ROW0_SURVIVE
# antigo (estes já valiam, só mudaram de nome):
#   DESTROY_WEIGHT = 5      -> W_DESTROY
#   EXTRA_BALL_WEIGHT = 30  -> W_EXTRA_BALL
#   BONUS_B_WEIGHT = 5      -> W_BONUS_B

W_HIT = 1.0  # por hit, já ponderado por ROW_MULTIPLIERS
W_DESTROY = 5.0  # por bloco destruído, já ponderado por ROW_MULTIPLIERS
W_EXTRA_BALL = 30.0  # por bola extra coletada
W_BONUS_B = 5.0  # por moeda B coletada (~2 hits em row 2)

# Estes dois começam DESLIGADOS: são regras que nunca rodaram de verdade, então
# entram em produção uma de cada vez, medindo o efeito nos turnos sobrevividos.
# W_ROW0_SURVIVE multiplica o HP que sobrou vivo na fileira crítica — é a única
# regra que olha o estado FINAL do tabuleiro, não o caminho da bola. Subir devagar
# (1..5): com ROW_MULTIPLIERS[0]=100, valor alto faz o bot ignorar o resto do jogo.
W_ROW0_SURVIVE = 0.0
W_LOOP = 0.0  # por grupo de 10 hits no mesmo voo ("loop" entre blocos)

# Multiplicadores por fileira (índice 0 = mais baixa, LEVELS[0]=480)
# antigo: 0:15, 1:5
ROW_MULTIPLIERS = {
    0: 100.0,  # GAME OVER IMINENTE: última fileira
    1: 50.0,  # PERIGO: penúltima fileira
    2: 25,  # MÉDIO: Terceira fileira (Y≈352)
    3: 15,  # NORMAL: Quarta fileira (Y≈289)
    4: 10,  # PADRÃO: Quinta fileira (Y≈225)
    5: 8,  # BAIXO: Sexta fileira (Y≈162)
    6: 5,  # DESPREZÍVEL: Fileira lá no teto (Y≈99)
}

# Substituído por ROW_MULTIPLIERS (scoring por fileira via block.row_index)
# DANGER_ZONE_Y = LEVELS[0]
# SCORE_DANGER_HIT = 20
# SCORE_DANGER_DESTROY = 500
# SCORE_CRITICAL_ROW = 2000

# ─────────────────────────────────────────────
# ÂNGULOS DE BUSCA
# ─────────────────────────────────────────────
ANGLE_MIN = 19  # graus (0 = direita pura, 90 = cima, 180 = esquerda)
ANGLE_MAX = 161  # simetrico: 180 - 19
# DECORATIVO — hoje só aparece num print em main.py. physics.find_best_angle usa
# `range(ANGLE_MIN, ANGLE_MAX + 1)`, com passo 1 chumbado, então mudar este valor
# não acelera nem afina a busca: muda só o texto do console. Ativar fica para depois.
ANGLE_STEP = 1  # resolução da busca (1° = 143 simulações)

# Escolha robusta: o ângulo vence pela média gaussiana do score dos vizinhos
# (±ceil(2σ) graus), não pelo pico isolado. σ = erro de execução que sobra depois
# da compensação de ANGLE_BIAS_TABELA (desvio medido no logs/bot_3: 1,2° direita, 1,5° esquerda).
# antigo: sem suavização (o pico de 1 grau ganhava)
ANGLE_ROBUSTO_SIGMA = 1.3

# ─────────────────────────────────────────────
# EXECUÇÃO / PYAUTOGUI
# ─────────────────────────────────────────────
# antigo: DRAG_DISTANCE = 150
DRAG_DISTANCE = px(150)  # distância do arrasto em pixels
DRAG_DURATION = 0.15  # segundos do gesto de arrasto

# Viés do disparo: o tiro sai puxado pra vertical (erro do "angulo real de
# saida" no log). O executor mira num ângulo que desconta esse viés. Se o log da
# próxima sessão mostrar erro médio longe de 0 numa faixa, reajuste o ponto dela.
# antigo: sem compensação (0.0 / 0.0)
# antigo (só logs/bot_3): ANGLE_BIAS_DIREITA = 1.6 / ANGLE_BIAS_ESQUERDA = -0.9
#   No logs/bot_4 sobrou +0,6° na direita, e na esquerda o viés cresce perto da
#   horizontal (0 até ~125°, -2° em 161°): uma constante errava pros dois lados.
# Viés (real - comandado) em função do ângulo COMANDADO, juntando bot_3 e bot_4
# (180 tiros, desvio ~1,2° por tiro). Interpolado linearmente entre os pontos e
# constante fora deles; o executor inverte a tabela pra mirar.
ANGLE_BIAS_TABELA = [
    (19, 1.9),
    (89, 1.9),
    (90, 0.0),
    (120, 0.0),
    (145, -1.1),
    (161, -2.0),
]
# removido: TURN_WAIT_TIME = 3.5
#   Era só o default de executor.wait_for_turn_end(), que ninguém chamava — o loop
#   real espera com polling de vision.detect_round_end(). Removido junto com
#   wait_for_turn_end/wait_for_turn_end_smart.

# Botão de acelerar
# antigo (absoluto): SPEED_UP_X = 1789 / SPEED_UP_Y = 958
SPEED_UP_X, SPEED_UP_Y = ponto_hud(437, 714)
SPEED_UP_DELAY = 0.3  # segundos após o disparo para clicar

# ─────────────────────────────────────────────
# DEBUG / VERBOSIDADE
# ─────────────────────────────────────────────
VISION_VERBOSE = False
