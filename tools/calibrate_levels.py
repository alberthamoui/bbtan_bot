"""
calibrate_levels.py — Ferramenta visual para ajustar limites de cada level.

Fase 1: Define o TOPO de cada level (7 linhas)
Fase 2: Define a BASE de cada level (7 linhas)

Controles:
  Seta Cima/Baixo    Troca level (L0-L6)
  Seta Esq/Dir       Ajuste fino ±1px
  Digitos + Enter    Define Y exato
  Backspace          Apaga ultimo digito
  Enter (sem digitos) Avanca para proxima fase / finaliza
  q                  Sair e imprimir valores
"""

import cv2
import numpy as np
import mss
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg

COLORS = [
    (0, 0, 255),  # Level 0 — vermelho
    (0, 128, 255),  # Level 1 — laranja
    (0, 255, 255),  # Level 2 — amarelo
    (0, 255, 0),  # Level 3 — verde
    (255, 255, 0),  # Level 4 — ciano
    (255, 128, 0),  # Level 5 — azul
    (255, 0, 255),  # Level 6 — magenta
]

rx, ry, rw, rh = cfg.GAME_REGION
half_block = cfg.BLOCK_SIZE // 2

tops = [c - half_block for c in cfg.LEVELS]
bots = [c + half_block for c in cfg.LEVELS]

sct = mss.MSS()


def run_phase(phase_name, values, fixed_values=None):
    """
    Loop de calibração para uma fase (topo ou base).
    fixed_values: linhas da outra fase para mostrar em cinza (referência).
    """
    selected = 0
    input_buffer = ""

    print(f"\n=== FASE: {phase_name.upper()} ===")
    print("Seta Cima/Baixo = troca level | Seta Esq/Dir = ±1px")
    print("Digitos + Enter = Y exato | Enter vazio = finalizar fase | q = sair\n")

    while True:
        monitor = {"top": ry, "left": rx, "width": rw, "height": rh}
        img = np.array(sct.grab(monitor), dtype=np.uint8)[:, :, :3].copy()

        # Linhas fixas da outra fase (cinza, referência)
        if fixed_values:
            for i, fy in enumerate(fixed_values):
                if 0 <= fy < rh:
                    for xs in range(0, rw, 12):
                        cv2.line(
                            img, (xs, fy), (min(xs + 6, rw), fy), (100, 100, 100), 1
                        )

        # Linhas da fase atual
        for i, y in enumerate(values):
            color = COLORS[i % len(COLORS)]
            thick = 3 if i == selected else 1
            cv2.line(img, (0, y), (rw, y), color, thick)
            cv2.putText(
                img,
                f"L{i} {phase_name}={y}",
                (5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # HUD
        hud = f"{phase_name.upper()} | L{selected} y={values[selected]}  |  Digitando: {input_buffer}_"
        cv2.putText(
            img, hud, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        cv2.imshow("Calibrar: Fileiras da grade", img)

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        if key == ord("q"):
            return False

        # Seta cima/baixo: troca level
        if key == 2490368:
            selected = (selected - 1) % len(values)
            input_buffer = ""
        elif key == 2621440:
            selected = (selected + 1) % len(values)
            input_buffer = ""

        # Seta dir/esq: ±1px
        elif key == 2555904:
            values[selected] = min(values[selected] + 1, rh)
        elif key == 2424832:
            values[selected] = max(values[selected] - 1, 0)

        # Digitos
        elif ord("0") <= key <= ord("9"):
            input_buffer += chr(key)

        # Enter
        elif key == 13:
            if input_buffer.isdigit():
                values[selected] = max(0, min(int(input_buffer), rh))
                input_buffer = ""
            else:
                return True

        # Backspace
        elif key == 8:
            input_buffer = input_buffer[:-1]


# Fase 1: topos
ok = run_phase("topo", tops)

if ok:
    # Fase 2: bases (mostra topos em cinza como referência)
    ok = run_phase("base", bots, fixed_values=tops)

print("\n=== VALORES EXTRAIDOS ===")
print("Copie e cole no config.py:\n")
print(f"LEVEL_TOPS = {tops}")
print(f"LEVEL_BOTS = {bots}")

cv2.destroyAllWindows()
