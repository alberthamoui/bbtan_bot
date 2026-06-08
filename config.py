"""
config.py — Constantes globais do BBTAN Bot.

Ajuste GAME_REGION para bater com a janela do seu emulador.
Use o script calibrate.py (incluso) para descobrir os valores certos.
"""

import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─────────────────────────────────────────────
# REGIÃO DO JOGO NA TELA (x, y, largura, altura)
# Rode calibrate.py para pegar esses valores
# ─────────────────────────────────────────────
GAME_REGION = (1353, 245, 451, 577)

# Margens internas da área jogável (pixels relativos à GAME_REGION)
WALL_LEFT = 5
WALL_RIGHT = GAME_REGION[2] - 5
WALL_TOP = 5
WALL_BOTTOM = GAME_REGION[3] - 10  # acima do robô/bola

# Y do centro de cada fileira, relativo ao frame capturado (0 = mais baixa/perigosa)
LEVELS = [480, 417, 352, 289, 225, 162, 99]


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


# --- BLOCOS (Se você precisar filtrar por cor específica) ---

# Bloco Amarelo (#ffe121)
BLOCK_YELLOW_HSV_LOW   = np.array([23, 180, 200])
BLOCK_YELLOW_HSV_HIGH  = np.array([29, 255, 255])

# Bloco Laranja (#ffad21)
BLOCK_ORANGE_HSV_LOW   = np.array([15, 180, 200])
BLOCK_ORANGE_HSV_HIGH  = np.array([22, 255, 255])

# Bloco Vermelho (#ff3b21)
BLOCK_RED_HSV_LOW      = np.array([0, 180, 200])
BLOCK_RED_HSV_HIGH     = np.array([6, 255, 255])

# Bloco Rosa (#ff2160)
BLOCK_PINK_HSV_LOW     = np.array([166, 180, 200])
BLOCK_PINK_HSV_HIGH    = np.array([174, 255, 255])

# Bloco Roxo (#8900ff)
BLOCK_PURPLE_HSV_LOW   = np.array([131, 180, 200])
BLOCK_PURPLE_HSV_HIGH  = np.array([141, 255, 255])

# BLOCOS — qualquer coisa colorida e brilhante (exclui fundo escuro)
BLOCK_HSV_LOW = np.array([0, 80, 100])
BLOCK_HSV_HIGH = np.array([179, 255, 255])

# Número de HP — texto branco (quase-branco) sobre fundo colorido
HP_TEXT_HSV_LOW  = np.array([0,   0,   252])
HP_TEXT_HSV_HIGH = np.array([175, 255, 255])
# Alternativa: texto escuro (roxo/azul) sobre fundo claro
# HP_TEXT_HSV_LOW  = np.array([129, 137, 218])
# HP_TEXT_HSV_HIGH = np.array([149, 255, 255])

# BOLA branca
BALL_HSV_LOW = np.array([0, 0, 200])
BALL_HSV_HIGH = np.array([179, 40, 255])

# Fundo escuro — tudo abaixo desse valor é ignorado
BG_VALUE_THRESHOLD = 80

# ─────────────────────────────────────────────
# DETECÇÃO DE BLOCOS
# ─────────────────────────────────────────────

# Grid do BBTAN: 7 colunas
GRID_COLUMNS = 7
BLOCK_SIZE = GAME_REGION[2] // GRID_COLUMNS  # pixels por bloco
MIN_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 0.15)  # ~15% da área de 1 bloco
MAX_BLOCK_AREA = int(BLOCK_SIZE * BLOCK_SIZE * 2.5)  # ~2.5 blocos (margem)
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

# Equação: h (bloco) = 2r (diâmetro da bola). Raio da bola = 4mm.
# Logo, o diâmetro da bola é igual ao tamanho do bloco (8mm).
# Portanto, o raio da bola é exatamente metade (50% ou 0.50) do BLOCK_SIZE.
BALL_RADIUS = max(3, int(BLOCK_SIZE * 0.50))

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

# Multiplicadores por fileira (índice 0 = mais baixa, LEVELS[0]=480)
ROW_MULTIPLIERS = {
    0: 15.0,  # CRÍTICO: Fileira colada no chão (Y≈480)
    1: 5.0,   # ALTO: Segunda fileira (Y≈417)
    2: 2.5,   # MÉDIO: Terceira fileira (Y≈352)
    3: 1.5,   # NORMAL: Quarta fileira (Y≈289)
    4: 1.0,   # PADRÃO: Quinta fileira (Y≈225)
    5: 0.8,   # BAIXO: Sexta fileira (Y≈162)
    6: 0.5    # DESPREZÍVEL: Fileira lá no teto (Y≈99)
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
TURN_WAIT_TIME = 3.5  # segundos entre turnos (ajustar conforme velocidade)
CONFIDENCE = 0.8  # confiança mínima para template matching (se usar)

# Botão de acelerar (coordenadas absolutas na tela)
SPEED_UP_X = 1789
SPEED_UP_Y = 958
SPEED_UP_DELAY = 0.3  # segundos após o disparo para clicar
