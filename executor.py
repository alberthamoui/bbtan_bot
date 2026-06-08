"""
executor.py — Converte decisões do bot em ações na tela via PyAutoGUI.

Responsabilidades:
  - Converter ângulo em coordenada de arrasto na tela
  - Executar o gesto de disparo (click-drag ou click)
  - Esperar o turno acabar
"""

import math
import time
import pyautogui

import config as cfg


# Segurança do PyAutoGUI
pyautogui.FAILSAFE = True   # mova o mouse pro canto superior esquerdo pra abortar
pyautogui.PAUSE = 0.05      # pausa mínima entre ações


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


def execute_shot_tap(ball_x: float, ball_y: float, angle_deg: float):
    """
    Método 1: Tap no ponto alvo.
    Alguns versões do BBTAN aceitam um tap simples na direção desejada.
    """
    sx, sy, tx, ty = angle_to_screen_coords(ball_x, ball_y, angle_deg)

    # Clica no ponto alvo (direção do disparo)
    pyautogui.click(int(tx), int(ty))


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
        button='left',
    )


def execute_shot_hold_release(ball_x: float, ball_y: float, angle_deg: float):
    """
    Método 3: Press-hold-move-release (simula toque contínuo).
    Mais compatível com emuladores Android que precisam de gesto longo.
    """
    sx, sy, tx, ty = angle_to_screen_coords(ball_x, ball_y, angle_deg)

    # Press na bola
    pyautogui.mouseDown(int(sx), int(sy), button='left')
    time.sleep(0.1)

    # Move suavemente até o alvo (define a direção)
    pyautogui.moveTo(int(tx), int(ty), duration=0.2)
    time.sleep(0.05)

    # Release para disparar
    pyautogui.mouseUp(button='left')


def speed_up_click():
    """Clica no botão de acelerar o jogo após o disparo."""
    time.sleep(cfg.SPEED_UP_DELAY)
    pyautogui.click(cfg.SPEED_UP_X, cfg.SPEED_UP_Y)


# Alias padrão — troque conforme o que funcionar no seu emulador
_base_shot = execute_shot_swipe


def execute_shot(ball_x: float, ball_y: float, angle_deg: float):
    """Dispara e clica no botão de acelerar."""
    _base_shot(ball_x, ball_y, angle_deg)
    speed_up_click()


def wait_for_turn_end(timeout: float = cfg.TURN_WAIT_TIME):
    """
    Espera o turno atual acabar.

    Versão simples: apenas espera um tempo fixo.
    Versão avançada (TODO): monitora a tela até as bolas pararem de se mover.
    """
    time.sleep(timeout)


def wait_for_turn_end_smart(ball_y_threshold: float = None):
    """
    Espera inteligente: monitora se a bola voltou para a base.
    Usa captura rápida para checar se ainda há movimento.
    """
    from vision import capture_screen, detect_ball

    if ball_y_threshold is None:
        ball_y_threshold = cfg.WALL_BOTTOM - 20

    check_interval = 0.3   # checa a cada 300ms
    stable_count = 0
    max_wait = 15.0         # timeout máximo de 15 segundos
    elapsed = 0.0

    while elapsed < max_wait:
        time.sleep(check_interval)
        elapsed += check_interval

        frame = capture_screen()
        ball_pos = detect_ball(frame)

        if ball_pos and ball_pos[1] > ball_y_threshold:
            stable_count += 1
            if stable_count >= 3:  # bola na base por 3 checks consecutivos
                time.sleep(0.5)   # pausa extra de segurança
                return
        else:
            stable_count = 0

    # Timeout — continua mesmo assim
    time.sleep(0.5)


if __name__ == "__main__":
    # Teste: mostra onde o bot clicaria para cada ângulo
    print("Teste de coordenadas de tiro:")
    for angle in [30, 60, 90, 120, 150]:
        sx, sy, tx, ty = angle_to_screen_coords(225, 550, angle)
        print(f"  {angle:3d}° -> start=({sx:.0f},{sy:.0f}) target=({tx:.0f},{ty:.0f})")