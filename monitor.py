"""
monitor.py — Monitor ao vivo do bot em janela pequena.

Atualiza continuamente mostrando:
  - Frame do jogo com blocos/bola detectados
  - Máscara HSV
  - Melhor trajetória simulada
  - Stats no overlay

Uso:
  python monitor.py
  python monitor.py --balls 5
  python monitor.py --fps 10       (mais rápido)
  python monitor.py --no-sim       (só detecção, sem simulação)
"""

import argparse
import time
import cv2
import numpy as np

import config as cfg
from vision import (
    capture_screen, detect_blocks, detect_powerups, detect_ball,
    BlockType,
)
from physics import find_best_angle, simulate_trajectory


WINDOW_NAME = "BBTAN Monitor"
SCALE = 0.5  # escala da janela (0.5 = metade do tamanho)


def build_frame(ball_count: int, run_sim: bool) -> np.ndarray:
    """Captura, detecta, simula e monta o frame de debug."""
    frame = capture_screen()
    h, w = frame.shape[:2]

    # ── Detecção ──
    blocks = detect_blocks(frame)
    powerups = detect_powerups(frame)
    ball_pos = detect_ball(frame)

    ball_x = ball_pos[0] if ball_pos else w / 2
    ball_y = ball_pos[1] if ball_pos else cfg.WALL_BOTTOM - 10

    # ── Painel esquerdo: detecções ──
    left = frame.copy()

    # Grid
    for i in range(1, cfg.GRID_COLUMNS):
        x = i * cfg.BLOCK_SIZE
        cv2.line(left, (x, 0), (x, h), (40, 40, 40), 1)

    # Paredes
    cv2.rectangle(left,
        (cfg.WALL_LEFT, cfg.WALL_TOP),
        (cfg.WALL_RIGHT, cfg.WALL_BOTTOM),
        (0, 80, 80), 1)

    # Blocos
    for b in blocks:
        color = (0, 255, 255) if b.block_type == BlockType.TRIANGLE else (0, 255, 0)
        cv2.rectangle(left, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 1)
        cv2.putText(left, f"{b.hp}", (b.x + 2, b.y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Power-ups
    for p in powerups:
        cv2.circle(left, (int(p.cx), int(p.cy)), int(p.radius), (255, 255, 0), 1)

    # Bola
    if ball_pos:
        cv2.circle(left, (int(ball_pos[0]), int(ball_pos[1])),
                   cfg.BALL_RADIUS, (255, 255, 255), 1)

    # ── Painel direito: simulação ──
    right = frame.copy()

    best = None
    sim_time = 0

    if run_sim and blocks:
        t0 = time.perf_counter()
        best, all_results = find_best_angle(
            ball_x, ball_y, blocks, powerups,
            ball_count=ball_count,
        )
        sim_time = time.perf_counter() - t0

        # Desenha top 5 trajetórias (cinza)
        for r in all_results[1:6]:
            tr = simulate_trajectory(
                ball_x, ball_y, r.angle,
                blocks, powerups,
                ball_count=ball_count,
                record_path=True,
            )
            if len(tr.path) >= 2:
                pts = np.array(tr.path, dtype=np.int32)
                cv2.polylines(right, [pts], False, (60, 60, 60), 1, cv2.LINE_AA)

        # Melhor em branco
        if len(best.path) >= 2:
            pts = np.array(best.path, dtype=np.int32)
            cv2.polylines(right, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

        # Seta
        if len(best.path) >= 5:
            dx = best.path[4][0] - ball_x
            dy = best.path[4][1] - ball_y
            cv2.arrowedLine(right,
                (int(ball_x), int(ball_y)),
                (int(ball_x + dx * 3), int(ball_y + dy * 3)),
                (0, 200, 255), 2, tipLength=0.3)

    # ── Combina ──
    combined = np.hstack([left, right])

    # ── Overlay de stats ──
    oy = 16
    lines = [
        f"Blocos: {len(blocks)}  PowerUps: {len(powerups)}  Bolas: {ball_count}",
    ]
    if ball_pos:
        lines.append(f"Bola: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")
    else:
        lines.append("Bola: NAO ENCONTRADA")

    if best:
        lines.append(f"Angulo: {best.angle:.0f}°  Score: {best.score:.0f}  "
                      f"Hits: {best.total_hits}  Destroy: {best.blocks_destroyed}  "
                      f"PwrUp: {best.powerups_collected}")
        lines.append(f"Sim: {sim_time*1000:.0f}ms")
    elif not run_sim:
        lines.append("Simulacao desligada (--no-sim)")

    for i, line in enumerate(lines):
        y = oy + i * 16
        cv2.putText(combined, line, (6, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(combined, line, (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1, cv2.LINE_AA)

    # Labels dos painéis
    cv2.putText(combined, "DETECCAO", (w - 80, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.putText(combined, "SIMULACAO", (w + w - 85, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Redimensiona
    if SCALE != 1.0:
        new_w = int(combined.shape[1] * SCALE)
        new_h = int(combined.shape[0] * SCALE)
        combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return combined


def main():
    parser = argparse.ArgumentParser(description="BBTAN Monitor ao vivo")
    parser.add_argument("--balls", type=int, default=1, help="Numero de bolas")
    parser.add_argument("--fps", type=int, default=5, help="Atualizacoes por segundo")
    parser.add_argument("--no-sim", action="store_true", help="Desliga simulacao (so deteccao)")
    parser.add_argument("--scale", type=float, default=0.5, help="Escala da janela (0.5 = metade)")
    args = parser.parse_args()

    global SCALE
    SCALE = args.scale

    interval = 1.0 / args.fps

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # always on top

    print(f"Monitor ao vivo — {args.fps} FPS, escala {args.scale}x")
    print(f"Bolas: {args.balls}  Simulacao: {'ON' if not args.no_sim else 'OFF'}")
    print(f"Pressione 'q' para fechar\n")

    while True:
        t0 = time.perf_counter()

        frame = build_frame(
            ball_count=args.balls,
            run_sim=not args.no_sim,
        )

        cv2.imshow(WINDOW_NAME, frame)

        elapsed = time.perf_counter() - t0
        wait_ms = max(1, int((interval - elapsed) * 1000))

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Monitor fechado.")


if __name__ == "__main__":
    main()
