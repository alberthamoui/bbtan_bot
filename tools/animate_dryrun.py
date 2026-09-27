"""
animate_dryrun.py — Visualizacao animada do dry-run.

Phase 1 (5s): desenha todos os angulos candidatos a partir da bola — cinza
semi-transparente, top-5 em amarelo, melhor em branco mais grosso.
Phase 2: anima as bolas percorrendo a melhor trajetoria, com trail crescente
em ~60 fps (~8px por frame) e marcadores vermelhos nas colisoes ao longo do
caminho. Cada bola segue o SEU path (SimResult.ball_paths): quando um bloco
quebra, as bolas seguintes seguem a trajetoria re-simulada sem ele.
Apos completar, congela 3s e volta pra Phase 1 em loop.

Pressione ESC ou 'q' para sair.
"""

import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from physics import find_best_angle, SimResult
from vision import GameState, scan_game

WINDOW = "Animacao: Dry-run da trajetoria"
TARGET_FPS = 60
FRAME_MS = max(1, int(1000 / TARGET_FPS))
PHASE1_DURATION = 5.0
FREEZE_DURATION = 3.0
CANDIDATE_LINE_LEN = 800  # pixels — generoso, cv2.line clipa fora do frame
# Espacamento (em indices do path) entre bolas consecutivas na Phase 2. Path eh
# sampleado a cada 4 steps de 2px = 8px por indice; BALL_SPACING=3 -> ~24px
# entre centros. Bola desenhada tem raio ~8, entao sobra folga visual entre elas.
BALL_SPACING = 3


def _angle_endpoint(
    ball_x: float, ball_y: float, angle_deg: float, length: int = CANDIDATE_LINE_LEN
) -> Tuple[int, int]:
    """Endpoint de uma reta vinda da bola na direcao do angulo (graus)."""
    rad = math.radians(angle_deg)
    return (
        int(ball_x + math.cos(rad) * length),
        int(ball_y - math.sin(rad) * length),
    )


def _put_label(img: np.ndarray, text: str, y: int = 25) -> None:
    """Texto branco com sombra preta no canto superior esquerdo."""
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _render_phase1(
    base: np.ndarray,
    ball_x: float,
    ball_y: float,
    all_results: List[SimResult],
    best: SimResult,
) -> np.ndarray:
    """Frame de Phase 1: leque de angulos candidatos."""
    overlay = base.copy()
    for r in all_results:
        ep = _angle_endpoint(ball_x, ball_y, r.angle)
        cv2.line(
            overlay, (int(ball_x), int(ball_y)), ep, (160, 160, 160), 1, cv2.LINE_AA
        )
    blended = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)

    top5 = all_results[:5]
    for r in top5:
        if r.angle == best.angle:
            continue
        ep = _angle_endpoint(ball_x, ball_y, r.angle)
        cv2.line(blended, (int(ball_x), int(ball_y)), ep, (0, 255, 255), 2, cv2.LINE_AA)

    ep = _angle_endpoint(ball_x, ball_y, best.angle)
    cv2.line(blended, (int(ball_x), int(ball_y)), ep, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.circle(
        blended, (int(ball_x), int(ball_y)), cfg.BALL_RADIUS, (255, 255, 255), -1
    )
    cv2.circle(blended, (int(ball_x), int(ball_y)), cfg.BALL_RADIUS, (0, 0, 0), 1)

    _put_label(
        blended,
        f"Phase 1 — {len(all_results)} angulos | top-5 amarelo | "
        f"melhor {best.angle:.0f}° (hits={best.total_hits})",
    )
    return blended


def _collision_path_indices(
    path: List[Tuple[float, float]], collisions: List[Tuple[float, float]]
) -> List[int]:
    """
    Para cada colisao, encontra o indice do path mais proximo.
    Permite revelar o marcador apenas quando a bola "chegou la".
    """
    if not path:
        return []
    indices = []
    for cx, cy in collisions:
        best_idx = 0
        best_d = float("inf")
        for i, (px, py) in enumerate(path):
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best_d:
                best_d = d
                best_idx = i
        indices.append(best_idx)
    return indices


def _render_phase2(
    base: np.ndarray,
    ball_paths: List[List[Tuple[float, float]]],
    ball_collisions: List[List[Tuple[float, float]]],
    path_col_indices: List[List[int]],
    current_idx: int,
    best: SimResult,
    ball_count: int = 1,
) -> np.ndarray:
    """
    Frame de Phase 2 com um path POR BOLA (SimResult.ball_paths): bolas
    posteriores seguem trajetorias re-simuladas apos os blocos destruidos
    pelas anteriores — um bloco quebrado para de refletir as bolas seguintes.

    A bola k esta em ball_paths[k][current_idx - k*BALL_SPACING]. Indice < 0
    ainda nao lancou; >= len(path) ja terminou (o trail dela fica completo).
    """
    frame = base.copy()

    n_balls = min(ball_count, len(ball_paths))

    # Progresso maximo por path unico — bolas que compartilham a mesma lista
    # (nada mais pra destruir) tem trail/marcadores desenhados uma vez so.
    progress: dict = {}  # id(path) -> [path, collisions, col_indices, max_idx]
    for k in range(n_balls):
        p = ball_paths[k]
        if not p:
            continue
        idx = current_idx - k * BALL_SPACING
        if idx < 0:
            continue
        idx = min(idx, len(p) - 1)
        entry = progress.get(id(p))
        if entry is None:
            progress[id(p)] = [p, ball_collisions[k], path_col_indices[k], idx]
        elif idx > entry[3]:
            entry[3] = idx

    total_cols = 0
    shown_cols = 0
    for p, cols, col_idx, max_idx in progress.values():
        if max_idx >= 1:
            pts = np.array(p[: max_idx + 1], dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)
        total_cols += len(cols)
        for (cx, cy), ci in zip(cols, col_idx):
            if ci <= max_idx:
                cv2.circle(frame, (int(cx), int(cy)), 7, (0, 0, 255), 2, cv2.LINE_AA)
                shown_cols += 1

    # Desenha ate ball_count bolas, cada uma no SEU path, com defasagem de
    # BALL_SPACING indices atras da anterior.
    balls_visible = 0
    ball_r = cfg.BALL_RADIUS
    for k in range(n_balls):
        p = ball_paths[k]
        idx = current_idx - k * BALL_SPACING
        if not p or idx < 0 or idx >= len(p):
            continue  # nao lancada ou ja terminou
        bx, by = p[idx]
        cv2.circle(frame, (int(bx), int(by)), ball_r, (255, 255, 255), -1)
        cv2.circle(frame, (int(bx), int(by)), ball_r, (0, 0, 0), 1)
        balls_visible += 1

    total = max(
        (
            len(p) - 1 + k * BALL_SPACING
            for k, p in enumerate(ball_paths[:n_balls])
            if p
        ),
        default=1,
    )
    _put_label(
        frame,
        f"Phase 2 — angulo {best.angle:.0f}° | passo {min(current_idx, total)}/{total} | "
        f"bolas {balls_visible}/{ball_count} | "
        f"colisoes {shown_cols}/{total_cols}",
    )
    return frame


def _wait(ms: int) -> bool:
    """waitKey que retorna True se o usuario quer sair."""
    key = cv2.waitKey(ms) & 0xFF
    return key in (27, ord("q"))


def _sleep_loop(window_frame: np.ndarray, seconds: float) -> bool:
    """Segura o frame na tela por N segundos, respondendo a teclas."""
    start = time.time()
    while time.time() - start < seconds:
        cv2.imshow(WINDOW, window_frame)
        if _wait(FRAME_MS):
            return True
    return False


def run_animate(state: GameState = None) -> None:
    """
    Captura (se nao receber state), simula, e roda o loop animado.
    Sai quando o usuario aperta ESC/q ou fecha a janela.
    """
    print("=" * 50)
    print("  BBTAN Bot — Dry Run Animado")
    print("=" * 50)
    print("  ESC ou 'q' para sair")
    print("=" * 50)

    if state is None:
        state = scan_game()

    if state.game_over:
        print(f"[!] Tela de game over detectada ({state.white_ratio:.0%} branca).")
        return
    if state.ball_x is None:
        print("[!] Bola nao encontrada — abortando.")
        return
    if not state.blocks:
        print("[!] Nenhum bloco detectado — abortando.")
        return

    ball_count = state.detected_ball_count if state.detected_ball_count > 0 else 1
    print(f"  Bola: ({state.ball_x:.0f}, {state.ball_y:.0f})  Bolas: {ball_count}")
    print(f"  Blocos: {len(state.blocks)}  Power-ups: {len(state.powerups)}")

    best, all_results = find_best_angle(
        state.ball_x,
        state.ball_y,
        state.blocks,
        state.powerups,
        ball_count=ball_count,
    )
    print(
        f"  Melhor angulo: {best.angle:.1f}°  "
        f"Hits: {best.total_hits}  Destruidos: {best.blocks_destroyed}  "
        f"Path: {len(best.path)} pts  Colisoes: {len(best.collisions)}"
    )

    base = state.debug_frame.copy()

    # Um path por bola (gravado pela simulacao exata com record_path=True).
    # Fallback: repete o path da 1a bola se ball_paths nao veio preenchido.
    ball_paths = list(best.ball_paths)
    ball_collisions = list(best.ball_collisions)
    if not ball_paths:
        ball_paths = [best.path] * ball_count
        ball_collisions = [best.collisions] * ball_count
    while len(ball_paths) < ball_count:
        ball_paths.append(ball_paths[-1])
        ball_collisions.append(ball_collisions[-1])

    # col_indices por path unico (bolas podem compartilhar a mesma lista)
    _col_idx_cache: dict = {}
    path_col_indices = []
    for p, c in zip(ball_paths, ball_collisions):
        if id(p) not in _col_idx_cache:
            _col_idx_cache[id(p)] = _collision_path_indices(p, c)
        path_col_indices.append(_col_idx_cache[id(p)])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            phase1_frame = _render_phase1(
                base, state.ball_x, state.ball_y, all_results, best
            )
            if _sleep_loop(phase1_frame, PHASE1_DURATION):
                return

            if best.path:
                # Estende a duracao pra que TODAS as bolas cheguem ao final:
                # bola k (path proprio) termina em len(path_k) + k*BALL_SPACING.
                total_frames = max(
                    len(p) + k * BALL_SPACING
                    for k, p in enumerate(ball_paths[:ball_count])
                )
                for i in range(total_frames):
                    frame = _render_phase2(
                        base,
                        ball_paths,
                        ball_collisions,
                        path_col_indices,
                        i,
                        best,
                        ball_count=ball_count,
                    )
                    cv2.imshow(WINDOW, frame)
                    if _wait(FRAME_MS):
                        return

                final = _render_phase2(
                    base,
                    ball_paths,
                    ball_collisions,
                    path_col_indices,
                    total_frames - 1,
                    best,
                    ball_count=ball_count,
                )
                if _sleep_loop(final, FREEZE_DURATION):
                    return
            else:
                # Sem path — mostra so a Phase 1 mais tempo e segue
                if _sleep_loop(phase1_frame, FREEZE_DURATION):
                    return
    finally:
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass


if __name__ == "__main__":
    run_animate()
