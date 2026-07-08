"""
config.py — Constantes globais do BBTAN Bot.

Ajuste GAME_REGION para bater com a janela do seu emulador.
Use o script calibrate.py (incluso) para descobrir os valores certos.
"""

import numpy as np

def printVisivel(x):
    print("\n")
    print("---"*20)
    print("\n")
    print(x)
    print("\n")
    print("---"*20)
    print("\n")

# ─────────────────────────────────────────────
# REGIÃO DO JOGO NA TELA (x, y, largura, altura)
# Rode calibrate.py para pegar esses valores
# ─────────────────────────────────────────────
# antigo: (1353, 245, 451, 577)
GAME_REGION = (1352, 244, 452, 578)

# Margens internas da área jogável (pixels relativos à GAME_REGION)
# antigo: WALL_LEFT = 5
WALL_LEFT = 0
# antigo: WALL_RIGHT = GAME_REGION[2] - 5  (=446)
WALL_RIGHT = 449
# antigo: WALL_TOP = 5
WALL_TOP = 2
# antigo: WALL_BOTTOM = GAME_REGION[3] - 10  (=567)  # acima do robô/bola
WALL_BOTTOM = 575

# Grid do BBTAN: 7 colunas
GRID_COLUMNS = 7
BLOCK_SIZE = GAME_REGION[2] // GRID_COLUMNS  # pixels por bloco

# Limites calibrados de cada coluna (via calibrate_columns.py)
COL_LEFTS  = [4, 67, 132, 195, 258, 321, 386]
COL_RIGHTS = [63, 128, 191, 255, 318, 382, 444]

# Limites de cada fileira, relativos ao frame capturado (0 = mais baixa/perigosa)
# Computados a partir de _GRID_TOP_Y e BLOCK_SIZE para cobertura contígua (sem gaps)
_GRID_TOP_Y = 67  # Y do topo da fileira mais alta (calibrar se mudar GAME_REGION)
LEVEL_TOPS = [_GRID_TOP_Y + (6 - i) * BLOCK_SIZE for i in range(7)]
LEVEL_BOTS = [_GRID_TOP_Y + (7 - i) * BLOCK_SIZE - 1 for i in range(7)]
# antigo: LEVEL_BOTS[0] = 514 (=_GRID_TOP_Y + 7*BLOCK_SIZE - 1). Cortava 1px
# do bloco da fileira mais baixa; ajustado pra bater exato com a base do bloco.
# antigo (tentativa anterior): LEVEL_BOTS[0] = WALL_BOTTOM = 575, mas esticava
# a fileira ate a base do playfield e fazia o codigo detectar o player como bloco.
LEVEL_BOTS[0] = 513
LEVELS = [(t + b) // 2 for t, b in zip(LEVEL_TOPS, LEVEL_BOTS)]


# ─────────────────────────────────────────────
# CORES HSV — extraídas da screenshot
# (OpenCV usa H: 0-179, S: 0-255, V: 0-255)
# ─────────────────────────────────────────────

# --- POWER-UPS ---

# Power-Up Laser Horiz/Vert (<>) — Azul (#008beb)
POWERUP_LASER_HSV_LOW  = np.array([97, 180, 180])
POWERUP_LASER_HSV_HIGH = np.array([107, 255, 255])

# Power-Up Bola Extra (+) — Verde Ciano (#00edd3)
POWERUP_EXTRA_BALL_HSV_LOW  = np.array([81, 180, 180])
POWERUP_EXTRA_BALL_HSV_HIGH = np.array([91, 255, 255])

# Power-Up Moeda/Bônus B (B) — Amarelo Ouro (#ffd539)
POWERUP_B_HSV_LOW      = np.array([20, 180, 200])
POWERUP_B_HSV_HIGH     = np.array([26, 255, 255])


# --- BLOCOS POR COR (usado para isolar texto de HP via inversão) ---

BLOCK_COLOR_RANGES = [
    # (nome, low, high)
    ("roxo",          np.array([126, 200, 150]), np.array([143, 255, 255])),
    ("vermelho",      np.array([0, 200, 150]),   np.array([9, 255, 255])),
    ("laranja",       np.array([16, 200, 150]),  np.array([22, 255, 255])),
    ("magenta",       np.array([144, 180, 120]), np.array([162, 255, 255])),
    ("amarelo_ouro",  np.array([22, 200, 150]),  np.array([28, 255, 255])),
    ("laranja_escuro", np.array([9, 200, 150]),  np.array([16, 255, 255])),
    ("rosa",           np.array([162, 200, 150]), np.array([179, 255, 255])),
    ("verde",          np.array([65, 180, 120]),  np.array([78, 255, 255])),
    ("amarelo_ouro",   np.array([22, 200, 150]),  np.array([28, 255, 255])),
    ("laranja2",       np.array([16, 200, 150]),  np.array([22, 255, 255])),
    ("azul",           np.array([108, 100, 150]), np.array([120, 255, 255])),   # #5c6ebf
    ("violeta",        np.array([118, 110, 160]), np.array([128, 255, 255])),   # #6557cf
    ("ciano",          np.array([82, 120, 130]),  np.array([92, 255, 255])),    # #42ab9f
]

# BLOCOS — qualquer coisa colorida (exclui fundo escuro V<50)
# BLOCK_HSV_LOW = np.array([0, 80, 100])  # antigo: perdia blocos escuros (roxo)
BLOCK_HSV_LOW = np.array([0, 60, 60])
BLOCK_HSV_HIGH = np.array([179, 255, 255])

# Número de HP — texto escuro sobre fundo colorido
# antigo (branco): [0, 0, 252] / [175, 255, 255]
HP_TEXT_HSV_LOW  = np.array([0, 0, 0])
HP_TEXT_HSV_HIGH = np.array([179, 255, 99])

# BOLA branca
BALL_HSV_LOW = np.array([0, 0, 200])
BALL_HSV_HIGH = np.array([179, 40, 255])

# Contagem de bolas (OCR) — região absoluta na tela onde aparece "x14", "x1", etc.
# antigo: (1324, 829, 483, 36)
BALL_COUNT_REGION = (1308, 824, 536, 33)
# antigo: [0, 0, 200] / [179, 41, 255]
BALL_COUNT_HSV_LOW  = np.array([0, 0, 140])
BALL_COUNT_HSV_HIGH = np.array([179, 50, 255])

# Indicador de fim de rodada — 4 regiões absolutas na tela com cor rosa (#e53572)
# Quando as 4 regiões contêm essa cor, a rodada acabou e o jogo pode ser analisado
ROUND_END_REGIONS = [
    (1445, 920, 22, 29),
    (1544, 919, 25, 30),
    (1649, 922, 22, 28),
    (1748, 919, 27, 32),
]
# #e53572 → RGB(229,53,114) → OpenCV HSV: H≈170, S≈196, V≈229
ROUND_END_HSV_LOW  = np.array([163, 150, 180])
ROUND_END_HSV_HIGH = np.array([177, 255, 255])

# Fundo escuro — tudo abaixo desse valor é ignorado
BG_VALUE_THRESHOLD = 80

# ─────────────────────────────────────────────
# DETECÇÃO DE BLOCOS
# ─────────────────────────────────────────────

MIN_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 0.15)  # ~15% da área de 1 bloco (triangulo ~50%)
MAX_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 4.5)  # bloco grande 2x2 (4x) + margem morfológica
TRIANGLE_VERTICES = 3
SQUARE_VERTICES = 4
POLY_APPROX_FACTOR = 0.04  # fator do approxPolyDP

# Tamanho esperado de um bloco na grid (pixels)
EXPECTED_BLOCK_SIZE = BLOCK_SIZE

# ─────────────────────────────────────────────
# SIMULAÇÃO DE FÍSICA
# ─────────────────────────────────────────────

# Bola de 4mm in-game. Tela mobile ~65mm de largura, 7 blocos.
# Bloco ~9.3mm, raio da bola = 2mm -> 2/9.3 ~= 0.22 do bloco
# BALL_RADIUS   = max(3, int(BLOCK_SIZE * 0.22))  # pixels

# Raio da bola = 1.77mm. Bloco = 8mm → fator = 1.77/8 ≈ 0.22
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.50))  # antigo: raio 4mm
BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.40))
# BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.22))

BALL_SPEED = 8.0  # pixels por step da simulação
MAX_SIM_STEPS = 2000  # limite de passos por simulação de ângulo
SIM_STEP_SIZE = 2.0  # resolução do step (menor = mais preciso, mais lento)

# ─────────────────────────────────────────────
# SCORING — pesos para a decisão de ângulo
# ─────────────────────────────────────────────
SCORE_HIT_BLOCK = 1  # cada hit em bloco
SCORE_DESTROY_BLOCK = 50  # bloco destruído (hp <= 0)
SCORE_HIT_POWERUP = 200  # coletar power-up (+)
SCORE_MULTI_BOUNCE = 5  # bônus por ricochete entre blocos consecutivos
SCORE_LOOP_BONUS = 100  # bônus se detectar "loop" entre blocos próximos
SCORE_ROW0_SURVIVE_PENALTY = 100  # penalidade por HP restante em blocos da row 0 após simulação

# Pesos para ranking (weighted_hits)
DESTROY_WEIGHT = 5      # bônus por bloco destruído (multiplicado por ROW_MULTIPLIER)
EXTRA_BALL_WEIGHT = 30  # bônus por bola extra coletada (abaixo de row 0/1, acima do resto)
BONUS_B_WEIGHT = 5      # bônus pequeno por moeda B coletada (~2 hits em row 2)

# Multiplicadores por fileira (índice 0 = mais baixa, LEVELS[0]=480)
# antigo: 0:15, 1:5
ROW_MULTIPLIERS = {
    0: 100.0,  # GAME OVER IMINENTE: última fileira
    1: 50.0,   # PERIGO: penúltima fileira
    2: 2.5,    # MÉDIO: Terceira fileira (Y≈352)
    3: 1.5,    # NORMAL: Quarta fileira (Y≈289)
    4: 1.0,    # PADRÃO: Quinta fileira (Y≈225)
    5: 0.8,    # BAIXO: Sexta fileira (Y≈162)
    6: 0.5     # DESPREZÍVEL: Fileira lá no teto (Y≈99)
}

# Substituído por ROW_MULTIPLIERS (scoring por fileira via block.row_index)
# DANGER_ZONE_Y = LEVELS[0]
# SCORE_DANGER_HIT = 20
# SCORE_DANGER_DESTROY = 500
# SCORE_CRITICAL_ROW = 2000

# ─────────────────────────────────────────────
# ÂNGULOS DE BUSCA
# ─────────────────────────────────────────────
ANGLE_MIN = 10  # graus (0 = direita pura, 90 = cima, 180 = esquerda)
ANGLE_MAX = 170
ANGLE_STEP = 1  # resolução da busca (1° = 160 simulações)

# ─────────────────────────────────────────────
# EXECUÇÃO / PYAUTOGUI
# ─────────────────────────────────────────────
DRAG_DISTANCE = 150  # distância do arrasto em pixels
DRAG_DURATION = 0.15  # segundos do gesto de arrasto
TURN_WAIT_TIME = 3.5   # segundos aguardando as bolas pararem após o disparo
PRESCAN_DELAY = 6.0    # segundos extras após OCR antes do próximo scan (blocos descendo)
CONFIDENCE = 0.8  # confiança mínima para template matching (se usar)

# Botão de acelerar (coordenadas absolutas na tela)
SPEED_UP_X = 1789
SPEED_UP_Y = 958
SPEED_UP_DELAY = 0.3  # segundos após o disparo para clicar
