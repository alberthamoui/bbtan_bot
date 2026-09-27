"""
calibrate_columns.py — Ferramenta visual para ajustar limites de cada coluna.

Cada coluna tem 2 linhas: borda esquerda e borda direita.
Há um mini gap entre a borda direita de uma coluna e a esquerda da próxima.

Fase 1: Define a borda ESQUERDA de cada coluna (7 linhas)
Fase 2: Define a borda DIREITA de cada coluna (7 linhas)

Controles:
  Seta Cima/Baixo    Troca coluna selecionada (C0-C6)
  Seta Esq/Dir       Ajuste fino ±1px
  Digitos + Enter    Define X exato
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
    (0, 0, 255),  # C0 — vermelho
    (0, 128, 255),  # C1 — laranja
    (0, 255, 255),  # C2 — amarelo
    (0, 255, 0),  # C3 — verde
    (255, 255, 0),  # C4 — ciano
    (255, 128, 0),  # C5 — azul
    (255, 0, 255),  # C6 — magenta
]

rx, ry, rw, rh = cfg.GAME_REGION

lefts = [cfg.WALL_LEFT + i * cfg.BLOCK_SIZE for i in range(cfg.GRID_COLUMNS)]
rights = [cfg.WALL_LEFT + (i + 1) * cfg.BLOCK_SIZE for i in range(cfg.GRID_COLUMNS)]

sct = mss.MSS()


def run_phase(phase_name, values, fixed_values=None):
    selected = 0
    input_buffer = ""

    print(f"\n=== FASE: {phase_name.upper()} ===")
    print("Seta Cima/Baixo = troca coluna | Seta Esq/Dir = ±1px")
    print("Digitos + Enter = X exato | Enter vazio = finalizar fase | q = sair\n")

    while True:
        monitor = {"top": ry, "left": rx, "width": rw, "height": rh}
        img = np.array(sct.grab(monitor), dtype=np.uint8)[:, :, :3].copy()

        # Linhas horizontais de referência (levels, cinza tracejado)
        for i in range(len(cfg.LEVEL_TOPS)):
            yt = cfg.LEVEL_TOPS[i]
            if 0 <= yt < rh:
                for xs in range(0, rw, 12):
                    cv2.line(img, (xs, yt), (min(xs + 6, rw), yt), (80, 80, 80), 1)

        # Linhas fixas da outra fase (cinza, referência)
        if fixed_values:
            for i, fx in enumerate(fixed_values):
                if 0 <= fx < rw:
                    for ys in range(0, rh, 12):
                        cv2.line(
                            img, (fx, ys), (fx, min(ys + 6, rh)), (100, 100, 100), 1
                        )

        # Linhas da fase atual
        for i, x in enumerate(values):
            color = COLORS[i % len(COLORS)]
            thick = 3 if i == selected else 1
            cv2.line(img, (x, 0), (x, rh), color, thick)
            label = f"C{i} {phase_name}={x}"
            cv2.putText(
                img,
                label,
                (x + 3, 15 + (i % 3) * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        # HUD
        hud = f"{phase_name.upper()} | C{selected} x={values[selected]}  |  Digitando: {input_buffer}_"
        cv2.putText(
            img, hud, (5, rh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

        cv2.imshow("Calibrar: Colunas da grade", img)

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        if key == ord("q"):
            return False

        if key == 2490368:
            selected = (selected - 1) % len(values)
            input_buffer = ""
        elif key == 2621440:
            selected = (selected + 1) % len(values)
            input_buffer = ""
        elif key == 2555904:
            values[selected] = min(values[selected] + 1, rw)
        elif key == 2424832:
            values[selected] = max(values[selected] - 1, 0)
        elif ord("0") <= key <= ord("9"):
            input_buffer += chr(key)
        elif key == 13:
            if input_buffer.isdigit():
                values[selected] = max(0, min(int(input_buffer), rw))
                input_buffer = ""
            else:
                return True
        elif key == 8:
            input_buffer = input_buffer[:-1]


# Fase 1: bordas esquerdas
ok = run_phase("esq", lefts)

if ok:
    # Fase 2: bordas direitas (mostra esquerdas em cinza como referência)
    ok = run_phase("dir", rights, fixed_values=lefts)

print("\n=== VALORES EXTRAIDOS ===")
print(f"COL_LEFTS  = {lefts}")
print(f"COL_RIGHTS = {rights}")
print(f"\nDetalhes:")
for i in range(cfg.GRID_COLUMNS):
    gap = ""
    if i < cfg.GRID_COLUMNS - 1:
        gap = f"  gap={lefts[i+1] - rights[i]}px"
    print(
        f"  Col {i}: esq={lefts[i]}  dir={rights[i]}  (largura={rights[i] - lefts[i]}px){gap}"
    )

cv2.destroyAllWindows()
