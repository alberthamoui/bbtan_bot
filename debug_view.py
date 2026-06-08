"""
debug_view.py — Visualização da trajetória simulada sobre o frame do jogo.

Útil para calibrar e verificar se a detecção e simulação estão funcionando.
Roda em modo standalone: captura, simula, desenha e mostra.
"""

import cv2
import numpy as np
from typing import List, Tuple

import config as cfg
from vision import GameState, scan_game, BlockType
from physics import SimResult, find_best_angle, simulate_trajectory


def draw_trajectory(
    frame: np.ndarray,
    path: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 200, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Desenha a trajetória simulada sobre o frame."""
    img = frame.copy()
    for i in range(1, len(path)):
        pt1 = (int(path[i - 1][0]), int(path[i - 1][1]))
        pt2 = (int(path[i][0]), int(path[i][1]))
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img


def draw_angle_heatmap(
    frame: np.ndarray,
    ball_x: float,
    ball_y: float,
    results: List[SimResult],
    top_n: int = 10,
) -> np.ndarray:
    """Desenha as top N trajetórias com cores do vermelho (pior) ao verde (melhor)."""
    img = frame.copy()

    if not results:
        return img

    top = results[:top_n]
    max_score = top[0].score if top[0].score > 0 else 1

    for i, result in enumerate(reversed(top)):
        # Re-simula com path
        r = simulate_trajectory(
            ball_x, ball_y, result.angle,
            [], [],  # sem blocos para rapidez
            record_path=True,
        )
        # Cor: verde (melhor) -> vermelho (pior)
        ratio = result.score / max_score
        color = (
            int(50 * (1 - ratio)),       # B
            int(255 * ratio),             # G
            int(255 * (1 - ratio)),       # R
        )
        img = draw_trajectory(img, r.path, color=color, thickness=1)

    # Desenha a melhor em branco mais grosso
    best = simulate_trajectory(
        ball_x, ball_y, top[0].angle,
        [], [],
        record_path=True,
    )
    img = draw_trajectory(img, best.path, color=(255, 255, 255), thickness=2)

    return img


def draw_stats_overlay(
    frame: np.ndarray,
    best: SimResult,
    state: GameState,
) -> np.ndarray:
    """Adiciona overlay com estatísticas no canto."""
    img = frame.copy()
    h, w = img.shape[:2]

    lines = [
        f"Angulo: {best.angle:.1f} graus",
        f"Score: {best.score:.0f}",
        f"Hits: {best.total_hits}",
        f"Destruidos: {best.blocks_destroyed}",
        f"Power-ups: {best.powerups_collected}",
        f"Blocos: {len(state.blocks)}",
    ]

    y_start = h - 20 * len(lines) - 10
    for i, line in enumerate(lines):
        y = y_start + i * 20
        # Sombra
        cv2.putText(img, line, (12, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # Texto
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def run_debug_view():
    """Modo debug interativo: captura, analisa, mostra resultado continuamente."""
    print("=== BBTAN Bot — Debug View ===")
    print("Pressione 'q' para sair, 'space' para re-capturar")
    print()

    while True:
        state = scan_game()
        print(f"Detectados: {len(state.blocks)} blocos, {len(state.powerups)} power-ups")

        if state.ball_x is None:
            print("AVISO: Bola nao encontrada! Usando posicao padrao.")
            ball_x = cfg.GAME_REGION[2] / 2
            ball_y = cfg.WALL_BOTTOM - 10
        else:
            ball_x = state.ball_x
            ball_y = state.ball_y
            print(f"Bola em: ({ball_x:.0f}, {ball_y:.0f})")

        if state.blocks:
            best, all_results = find_best_angle(
                ball_x, ball_y,
                state.blocks, state.powerups,
            )
            print(f"Melhor: {best.angle:.1f}° (score={best.score:.0f}, hits={best.total_hits})")

            # Desenha visualização
            vis = state.debug_frame.copy()
            vis = draw_trajectory(vis, best.path, color=(0, 200, 255), thickness=2)
            vis = draw_stats_overlay(vis, best, state)
        else:
            vis = state.debug_frame.copy()
            print("Nenhum bloco detectado.")

        cv2.imshow("BBTAN Bot Debug", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_debug_view()