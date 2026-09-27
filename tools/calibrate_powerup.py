"""
calibrate_powerup.py — Mede o raio de power-ups na tela do jogo.

Clique no centro de um power-up e arraste até a borda.
O raio em pixels é mostrado em tempo real.

Controles:
  Clique + arraste    Mede raio do centro até o mouse
  r                   Limpa medições
  Enter               Imprime valores e sai
  q                   Sair
"""

import cv2
import numpy as np
import mss
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg

rx, ry, rw, rh = cfg.GAME_REGION
sct = mss.MSS()

center = None
radius = 0
dragging = False
measurements = []


def on_mouse(event, x, y, flags, param):
    global center, radius, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        radius = 0
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - center[0]
        dy = y - center[1]
        radius = int(np.sqrt(dx * dx + dy * dy))
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        if radius > 2:
            area = np.pi * radius * radius
            measurements.append((center, radius, area))
            print(
                f"  Medicao {len(measurements)}: centro=({center[0]},{center[1]})  raio={radius}px  diametro={radius*2}px  area={area:.0f}px²"
            )


cv2.namedWindow("Calibrar: Raio do power-up")
cv2.setMouseCallback("Calibrar: Raio do power-up", on_mouse)

print("\n=== MEDIR POWER-UP ===")
print("Clique no centro do power-up e arraste ate a borda")
print("r = limpar | Enter = finalizar | q = sair\n")

while True:
    monitor = {"top": ry, "left": rx, "width": rw, "height": rh}
    img = np.array(sct.grab(monitor), dtype=np.uint8)[:, :, :3].copy()

    # Medições anteriores (cinza)
    for c, r, a in measurements:
        cv2.circle(img, c, r, (150, 150, 150), 1)
        cv2.putText(
            img,
            f"r={r}",
            (c[0] + r + 3, c[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (150, 150, 150),
            1,
        )

    # Medição atual (ciano)
    if center and (dragging or radius > 2):
        cv2.circle(img, center, radius, (255, 255, 0), 2)
        cv2.circle(img, center, 2, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"r={radius}  d={radius*2}  area={np.pi*radius*radius:.0f}",
            (center[0] + radius + 5, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

    # HUD
    hud = f"Medicoes: {len(measurements)}"
    cv2.putText(
        img, hud, (5, rh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )

    cv2.imshow("Calibrar: Raio do power-up", img)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 13:
        break
    elif key == ord("r"):
        measurements.clear()
        center = None
        radius = 0
        print("  Medicoes limpas.")

if measurements:
    radii = [r for _, r, _ in measurements]
    avg_r = sum(radii) / len(radii)
    avg_area = np.pi * avg_r * avg_r
    print(f"\n=== RESULTADO ===")
    print(f"Medicoes: {len(measurements)}")
    print(f"Raios: {radii}")
    print(f"Raio medio: {avg_r:.1f}px")
    print(f"Diametro medio: {avg_r*2:.1f}px")
    print(f"Area media: {avg_area:.0f}px²")
else:
    print("\nNenhuma medicao feita.")

cv2.destroyAllWindows()
