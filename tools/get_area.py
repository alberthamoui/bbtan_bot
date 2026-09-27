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

NUM_AREAS = 7


def calibrate_region():
    """Captura tela inteira e permite selecionar 7 regiões."""
    print("Posicione a janela do emulador e pressione Enter...")
    input()

    sct = mss.mss()
    monitor = sct.monitors[1]  # monitor principal
    raw = sct.grab(monitor)
    frame = np.array(raw, dtype=np.uint8)[:, :, :3]

    print(f"Selecione {NUM_AREAS} areas com o mouse (clique e arraste).")
    print("Pressione Enter para confirmar cada area, C para cancelar.")

    areas = []
    for i in range(NUM_AREAS):
        title = f"Selecionar area {i + 1} de {NUM_AREAS}"
        print(f"\n[{i + 1}/{NUM_AREAS}] {title}")
        roi = cv2.selectROI(title, frame, fromCenter=False)
        cv2.destroyWindow(title)

        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = [int(v) for v in roi]
            areas.append((x, y, w, h))
            print(f"  -> x={x}, y={y}, w={w}, h={h}")
        else:
            print("  -> Cancelado. Encerrando.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    print(f"\n>>> {NUM_AREAS} areas selecionadas:")
    for i, (x, y, w, h) in enumerate(areas):
        print(f"  Area {i + 1}: x={x}, y={y}, w={w}, h={h}")


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
    else:
        print(f"Modo desconhecido: {mode}")
        print("Use: region, colors, ou test")
