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


def execute_shot_swipe(ball_x: float, ball_y: float, angle_deg: float):
    """
    Método 2: Swipe/drag da bola até o ponto alvo.
    Este é o método mais confiável para a maioria dos emuladores.
    """
    sx, sy, tx, ty = angle_to_screen_coords(ball_x, ball_y, angle_deg)

    # Move para a posição da bola
    pyautogui.moveTo(int(sx), int(sy), duration=0.05)
    time.sleep(0.05)

    # Arrasta na direção do disparo
    pyautogui.drag(
        int(tx - sx),
        int(ty - sy),
        duration=cfg.DRAG_DURATION,
        button="left",
    )


def speed_up_click():
    """Clica no botão de acelerar o jogo após o disparo."""
    time.sleep(cfg.SPEED_UP_DELAY)
    pyautogui.click(cfg.SPEED_UP_X, cfg.SPEED_UP_Y)


def click_retrieve_button():
    """Clica no centro do botão retrieve pra recolher as bolas e encerrar a rodada."""
    rx, ry, rw, rh = cfg.RETRIEVE_BUTTON_REGION
    pyautogui.click(rx + rw // 2, ry + rh // 2)


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
