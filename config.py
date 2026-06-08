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
GAME_REGION = (1353, 245, 451, 577)

# Margens internas da área jogável (pixels relativos à GAME_REGION)
WALL_LEFT = 5
WALL_RIGHT = GAME_REGION[2] - 5
WALL_TOP = 5
WALL_BOTTOM = GAME_REGION[3] - 10  # acima do robô/bola

LEVELS = [723, 661, 596, 529, 468, 406, 341]


# ─────────────────────────────────────────────
# CORES HSV — extraídas da screenshot
# (OpenCV usa H: 0-179, S: 0-255, V: 0-255)
# ─────────────────────────────────────────────

# BLOCOS — qualquer coisa colorida e brilhante (exclui fundo escuro)
BLOCK_HSV_LOW = np.array([0, 80, 100])
BLOCK_HSV_HIGH = np.array([179, 255, 255])

# POWER-UPS ciano (+)
POWERUP_HSV_LOW = np.array([80, 100, 150])
POWERUP_HSV_HIGH = np.array([100, 255, 255])

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

# Multiplicadores para cada fileira de LEVELS = [723, 661, 596, 529, 468, 406, 341]
# Index 0 (723) é a última fileira antes do Game Over, ganha peso máximo.
ROW_MULTIPLIERS = {
    0: 15.0,  # CRÍTICO: Fileira colada no chão (Y=723)
    1: 5.0,   # ALTO: Segunda fileira de baixo para cima (Y=661)
    2: 2.5,   # MÉDIO: Terceira fileira (Y=596)
    3: 1.5,   # NORMAL: Quarta fileira (Y=529)
    4: 1.0,   # PADRÃO: Quinta fileira (Y=468)
    5: 0.8,   # BAIXO: Sexta fileira (Y=406)
    6: 0.5    # DESPREZÍVEL: Fileira lá no teto (Y=341)
}

# Zona de perigo — blocos perto do fundo
# DANGER_ZONE_Y: a partir de qual Y (relativo à GAME_REGION) o bloco é perigoso
# Quanto mais embaixo, maior o multiplicador
DANGER_ZONE_Y = LEVELS[0]  # últimos 40% da tela
SCORE_DANGER_HIT = 20  # bônus por hit em bloco na zona de perigo
SCORE_DANGER_DESTROY = 500  # bônus ALTO por destruir bloco na zona de perigo
SCORE_CRITICAL_ROW = 2000  # bônus se bloco está na última fileira (quase game over)

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
