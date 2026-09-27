"""
calibrate.py — Utilitário de calibração para o BBTAN Bot.

Funcionalidades:
  1. Captura a tela inteira e deixa você selecionar a região do jogo
  2. Mostra sliders HSV para ajustar as máscaras de cor em tempo real
  3. Testa detecção de blocos/bola no frame atual

Uso:
  python calibrate.py region    -> Selecionar região do jogo
  python calibrate.py colors    -> Ajustar filtros HSV
  python calibrate.py test      -> Testar detecção com config atual
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
import mss
from vision import scan_game, show_debug


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config as cfg


def calibrate_region():
    """Captura tela inteira e permite selecionar a região do jogo."""
    print("Posicione a janela do emulador e pressione Enter...")
    input()

    sct = mss.mss()
    monitor = sct.monitors[1]  # monitor principal
    raw = sct.grab(monitor)
    frame = np.array(raw, dtype=np.uint8)[:, :, :3]

    print("Selecione a regiao do jogo com o mouse (clique e arraste).")
    print("Pressione Enter para confirmar, C para cancelar.")

    roi = cv2.selectROI("Selecione a Regiao do Jogo", frame, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = [int(v) for v in roi]
        print(f"\n>>> Atualize config.py com:")
        print(f"GAME_REGION = ({x}, {y}, {w}, {h})")
    else:
        print("Selecao cancelada.")


def calibrate_colors():
    """Interface com trackbars para ajustar filtros HSV em tempo real."""
    print("Ajuste os sliders HSV. Pressione 'q' para sair.")
    print("Os valores finais serao impressos no terminal.\n")

    sct = mss.mss()
    rx, ry, rw, rh = cfg.GAME_REGION
    monitor = {"top": ry, "left": rx, "width": rw, "height": rh}

    cv2.namedWindow("Calibrar HSV: Controles", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Calibrar HSV: Mask resultante", cv2.WINDOW_NORMAL)

    # Trackbars com valores iniciais dos blocos laranja
    cv2.createTrackbar(
        "H Low",
        "Calibrar HSV: Controles",
        int(cfg.BLOCK_HSV_LOW[0]),
        179,
        lambda x: None,
    )
    cv2.createTrackbar(
        "S Low",
        "Calibrar HSV: Controles",
        int(cfg.BLOCK_HSV_LOW[1]),
        255,
        lambda x: None,
    )
    cv2.createTrackbar(
        "V Low",
        "Calibrar HSV: Controles",
        int(cfg.BLOCK_HSV_LOW[2]),
        255,
        lambda x: None,
    )
    cv2.createTrackbar(
        "H High",
        "Calibrar HSV: Controles",
        int(cfg.BLOCK_HSV_HIGH[0]),
        179,
        lambda x: None,
    )
    cv2.createTrackbar(
        "S High",
        "Calibrar HSV: Controles",
        int(cfg.BLOCK_HSV_HIGH[1]),
        255,
        lambda x: None,
    )
    cv2.createTrackbar(
        "V High",
        "Calibrar HSV: Controles",
        int(cfg.BLOCK_HSV_HIGH[2]),
        255,
        lambda x: None,
    )

    while True:
        raw = sct.grab(monitor)
        frame = np.array(raw, dtype=np.uint8)[:, :, :3]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_lo = cv2.getTrackbarPos("H Low", "Calibrar HSV: Controles")
        s_lo = cv2.getTrackbarPos("S Low", "Calibrar HSV: Controles")
        v_lo = cv2.getTrackbarPos("V Low", "Calibrar HSV: Controles")
        h_hi = cv2.getTrackbarPos("H High", "Calibrar HSV: Controles")
        s_hi = cv2.getTrackbarPos("S High", "Calibrar HSV: Controles")
        v_hi = cv2.getTrackbarPos("V High", "Calibrar HSV: Controles")

        low = np.array([h_lo, s_lo, v_lo])
        high = np.array([h_hi, s_hi, v_hi])

        mask = cv2.inRange(hsv, low, high)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Encontra contornos na máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = frame.copy()
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > cfg.MIN_BLOCK_AREA:
                cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 2)
                count += 1

        cv2.putText(
            debug,
            f"Contornos: {count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Calibrar HSV: Controles", debug)
        cv2.imshow("Calibrar HSV: Mask resultante", result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print(f"\n>>> Valores HSV finais:")
            print(f"HSV_LOW  = np.array([{h_lo}, {s_lo}, {v_lo}])")
            print(f"HSV_HIGH = np.array([{h_hi}, {s_hi}, {v_hi}])")
            break

    cv2.destroyAllWindows()


def test_detection():
    """Testa a detecção com a configuração atual."""
    print("Capturando e analisando...")
    state = scan_game()

    print(f"\nResultados:")
    print(f"  Blocos detectados: {len(state.blocks)}")
    for b in state.blocks:
        print(
            f"    {b.block_type.value:10s} @ ({b.x:3d},{b.y:3d}) "
            f"{b.w}x{b.h} hp={b.hp} orient={b.tri_orientation.value}"
        )

    print(f"  Power-ups: {len(state.powerups)}")
    for p in state.powerups:
        print(f"    (+) @ ({p.cx:.0f},{p.cy:.0f}) r={p.radius:.0f}")

    if state.ball_x:
        print(f"  Bola: ({state.ball_x:.0f}, {state.ball_y:.0f})")
    else:
        print(f"  Bola: NAO ENCONTRADA")

    print("\nMostrando frame de debug (pressione qualquer tecla)...")
    show_debug(state, wait=0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python calibrate.py [region|colors|test]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "region":
        calibrate_region()
    elif mode == "colors":
        calibrate_colors()
    elif mode == "test":
        test_detection()
    else:
        print(f"Modo desconhecido: {mode}")
        print("Use: region, colors, ou test")
