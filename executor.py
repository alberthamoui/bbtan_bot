"""
executor.py — Converte decisões do bot em ações na tela via PyAutoGUI.

Responsabilidades:
  - Converter ângulo em coordenada de arrasto na tela
  - Executar o gesto de disparo (click-drag ou click)

A espera pelo fim do turno NÃO mora aqui — é feita no loop de main.py com
polling de vision.detect_round_end().
"""

import math
import time
import pyautogui

import config as cfg
import tela

# Segurança do PyAutoGUI
pyautogui.FAILSAFE = True  # mova o mouse pro canto superior esquerdo pra abortar
pyautogui.PAUSE = 0.05  # pausa mínima entre ações


def angle_to_screen_coords(
    ball_x: float,
    ball_y: float,
    angle_deg: float,
    drag_distance: float = cfg.DRAG_DISTANCE,
) -> tuple:
    """
    Converte posição da bola + ângulo em coordenadas absolutas de tela
    para o ponto de início e fim do arrasto.

    No BBTAN, o disparo funciona por swipe:
    - Você toca na bola e arrasta na direção OPOSTA ao disparo
    - Ou simplesmente toca no ponto acima da bola na direção desejada

    Retorna: (start_x, start_y, end_x, end_y) em coords absolutas de tela.
    """
    gx, gy, gw, gh = cfg.GAME_REGION

    # Posição absoluta da bola na tela
    abs_ball_x = gx + ball_x
    abs_ball_y = gy + ball_y

    # Direção do disparo (a bola vai NESSA direção)
    angle_rad = math.radians(angle_deg)
    dir_x = math.cos(angle_rad)
    dir_y = -math.sin(angle_rad)  # Y invertido (tela)

    # O ponto alvo é NA DIREÇÃO do disparo (acima da bola)
    # Alguns emuladores usam tap no ponto alvo, outros usam swipe
    target_x = abs_ball_x + dir_x * drag_distance
    target_y = abs_ball_y + dir_y * drag_distance

    return (abs_ball_x, abs_ball_y, target_x, target_y)


def _vies(comandado: float) -> float:
    """Viés (real - comandado) interpolado de cfg.ANGLE_BIAS_TABELA."""
    tabela = cfg.ANGLE_BIAS_TABELA
    if comandado <= tabela[0][0]:
        return tabela[0][1]
    for (a0, b0), (a1, b1) in zip(tabela, tabela[1:]):
        if comandado <= a1:
            return b0 + (b1 - b0) * (comandado - a0) / (a1 - a0)
    return tabela[-1][1]


def angulo_compensado(angle_deg: float) -> float:
    """
    Ângulo em que mirar pra bola sair em `angle_deg`: o comandado c com
    c + viés(c) = angle_deg. A tabela é por ângulo comandado, então resolve por
    ponto fixo (c = alvo - viés(c)); 3 passadas bastam, o viés quase não muda
    em 2 graus.
    """
    # antigo: angle_deg - ANGLE_BIAS_DIREITA (< 90) / - ANGLE_BIAS_ESQUERDA (> 90)
    c = angle_deg
    for _ in range(3):
        c = angle_deg - _vies(c)
    return c


def execute_shot_swipe(ball_x: float, ball_y: float, angle_deg: float):
    """
    Método 2: Swipe/drag da bola até o ponto alvo.
    Este é o método mais confiável para a maioria dos emuladores.
    """
    # antigo: angle_to_screen_coords(ball_x, ball_y, angle_deg) — sem compensar o viés
    sx, sy, tx, ty = angle_to_screen_coords(ball_x, ball_y, angulo_compensado(angle_deg))

    # Move para a posição da bola (antigo: pyautogui.moveTo / pyautogui.drag
    # direto; agora pelo tela.py, que o jogo local troca)
    tela.mover(int(sx), int(sy), duracao=0.05)
    time.sleep(0.05)

    # Arrasta na direção do disparo
    tela.arrastar(int(tx - sx), int(ty - sy), duracao=cfg.DRAG_DURATION)


def click_acelerar():
    """Clica no botão de acelerar, sem esperar (o modo --referencia controla o atraso)."""
    tela.clicar(cfg.SPEED_UP_X, cfg.SPEED_UP_Y)  # antigo: pyautogui.click


def speed_up_click():
    """Clica no botão de acelerar o jogo após o disparo."""
    time.sleep(cfg.SPEED_UP_DELAY)
    # antigo: pyautogui.click(cfg.SPEED_UP_X, cfg.SPEED_UP_Y)
    click_acelerar()


def click_retrieve_button():
    """Clica no botão retrieve pra recolher as bolas e encerrar a rodada."""
    # antigo: centro de RETRIEVE_BUTTON_REGION (1577, 966)
    tela.clicar(*cfg.RETRIEVE_CLICK)  # antigo: pyautogui.click


# Alias padrão — troque conforme o que funcionar no seu emulador
_base_shot = execute_shot_swipe


def execute_shot(ball_x: float, ball_y: float, angle_deg: float):
    """Dispara e clica no botão de acelerar."""
    _base_shot(ball_x, ball_y, angle_deg)
    speed_up_click()


# removido: wait_for_turn_end() e wait_for_turn_end_smart()
#   Nenhuma das duas era chamada. O loop de main.py espera o turno acabar com
#   polling de vision.detect_round_end() (as 4 regiões rosa). Junto com elas saiu
#   o cfg.TURN_WAIT_TIME, que só existia como default da primeira.
