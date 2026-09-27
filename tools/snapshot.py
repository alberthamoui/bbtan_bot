"""
snapshot.py — Visualização completa de 1 turno do bot.

Mostra TUDO que o bot vê e pensa:
  1. Frame cru capturado
  2. Máscara HSV (o que o filtro de cor isolou)
  3. Blocos detectados com tipo, posição e HP
  4. Power-ups detectados
  5. Posição da bola
  6. Heatmap de todos os ângulos testados
  7. Trajetória do melhor ângulo
  8. Ranking dos top 10 ângulos

Salva tudo como imagens + print detalhado no terminal.

Uso:
  python snapshot.py
  python snapshot.py --balls 5
  python snapshot.py --save    (salva PNGs em vez de mostrar janelas)
"""

import argparse
import time
import cv2
import numpy as np
from typing import List

import config as cfg
from vision import (
    capture_screen,
    detect_blocks,
    detect_powerups,
    detect_ball,
    Block,
    BlockType,
    PowerUp,
)
from physics import simulate_trajectory, find_best_angle, SimResult


def draw_grid(img: np.ndarray) -> np.ndarray:
    """Desenha a grade de 7 colunas sobre o frame."""
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, cfg.GRID_COLUMNS):
        x = i * cfg.BLOCK_SIZE
        cv2.line(out, (x, 0), (x, h), (40, 40, 40), 1)
    # Paredes
    cv2.rectangle(
        out,
        (cfg.WALL_LEFT, cfg.WALL_TOP),
        (cfg.WALL_RIGHT, cfg.WALL_BOTTOM),
        (0, 100, 100),
        1,
    )
    return out


def draw_detections(
    img: np.ndarray, blocks: List[Block], powerups: List[PowerUp], ball_pos
) -> np.ndarray:
    """Desenha todos os objetos detectados com labels detalhados."""
    out = img.copy()

    for i, b in enumerate(blocks):
        if b.block_type == BlockType.TRIANGLE:
            color = (0, 255, 255)  # amarelo
            label = f"TRI hp={b.hp}"
            # Desenha o triângulo detectado
            if len(b.vertices) >= 3:
                pts = b.vertices.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(out, [pts], True, color, 2)
        else:
            color = (0, 255, 0)  # verde
            label = f"SQ hp={b.hp}"

        # Bounding box
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)

        # Label com indice
        text = f"#{i} {label} ({b.w}x{b.h})"
        cv2.putText(
            out,
            text,
            (b.x, b.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

        # Centro
        cv2.circle(out, (int(b.cx), int(b.cy)), 3, color, -1)

    for i, p in enumerate(powerups):
        cv2.circle(out, (int(p.cx), int(p.cy)), int(p.radius), (255, 255, 0), 2)
        cv2.putText(
            out,
            f"+{i}",
            (int(p.cx) - 8, int(p.cy) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if ball_pos:
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(out, (bx, by), cfg.BALL_RADIUS, (255, 255, 255), 2)
        cv2.circle(out, (bx, by), 2, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"BOLA ({bx},{by})",
            (bx + 12, by),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_all_trajectories(
    img: np.ndarray,
    ball_x: float,
    ball_y: float,
    all_results: List[SimResult],
    blocks: List[Block],
    powerups: List[PowerUp],
    ball_count: int,
) -> np.ndarray:
    """Heatmap: todas as trajetórias coloridas por score."""
    out = img.copy()

    if not all_results:
        return out

    max_score = max(r.score for r in all_results) if all_results[0].score > 0 else 1
    min_score = min(r.score for r in all_results)
    score_range = max(max_score - min_score, 1)

    # Desenha do pior pro melhor (melhor fica por cima)
    sorted_results = sorted(all_results, key=lambda r: r.score)

    for result in sorted_results:
        # Re-simula com path
        r = simulate_trajectory(
            ball_x,
            ball_y,
            result.angle,
            blocks,
            powerups,
            ball_count=ball_count,
            record_path=True,
        )
        if len(r.path) < 2:
            continue

        # Cor: vermelho (ruim) -> verde (bom)
        ratio = (result.score - min_score) / score_range
        color = (
            0,
            int(255 * ratio),
            int(255 * (1 - ratio)),
        )

        pts = np.array(r.path, dtype=np.int32)
        cv2.polylines(out, [pts], False, color, 1, cv2.LINE_AA)

    return out


def draw_best_trajectory(
    img: np.ndarray, best: SimResult, ball_x: float, ball_y: float
) -> np.ndarray:
    """Desenha a melhor trajetória em destaque com marcadores de colisão."""
    out = img.copy()

    if len(best.path) < 2:
        return out

    # Linha branca grossa
    pts = np.array(best.path, dtype=np.int32)
    cv2.polylines(out, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

    # Ponto de inicio
    cv2.circle(out, (int(ball_x), int(ball_y)), 6, (0, 200, 255), -1)

    # Seta indicando direção inicial
    if len(best.path) >= 5:
        dx = best.path[4][0] - ball_x
        dy = best.path[4][1] - ball_y
        cv2.arrowedLine(
            out,
            (int(ball_x), int(ball_y)),
            (int(ball_x + dx * 3), int(ball_y + dy * 3)),
            (0, 200, 255),
            2,
            tipLength=0.3,
        )

    return out


def draw_score_bar(img: np.ndarray, all_results: List[SimResult]) -> np.ndarray:
    """Desenha um gráfico de barras dos scores por ângulo na lateral."""
    h, w = img.shape[:2]
    bar_w = 300
    bar_h = h

    bar_img = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    bar_img[:] = (30, 30, 30)

    if not all_results:
        return np.hstack([img, bar_img])

    max_score = max(r.score for r in all_results) if all_results[0].score > 0 else 1
    sorted_by_angle = sorted(all_results, key=lambda r: r.angle)
    n = len(sorted_by_angle)

    # Título
    cv2.putText(
        bar_img,
        "SCORE POR ANGULO",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Barras
    y_start = 35
    y_space = max(1, (bar_h - y_start - 10) // n)

    for i, r in enumerate(sorted_by_angle):
        y = y_start + i * y_space
        if y + y_space > bar_h - 5:
            break

        ratio = r.score / max_score if max_score > 0 else 0
        bar_len = int(ratio * (bar_w - 60))

        color = (0, int(255 * ratio), int(255 * (1 - ratio)))
        cv2.rectangle(
            bar_img, (50, y), (50 + bar_len, y + max(1, y_space - 1)), color, -1
        )

        # Label a cada 10 graus
        if int(r.angle) % 10 == 0:
            cv2.putText(
                bar_img,
                f"{int(r.angle)}°",
                (2, y + y_space),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (150, 150, 150),
                1,
            )

    return np.hstack([img, bar_img])


def print_full_report(
    blocks, powerups, ball_pos, best, all_results, ball_count, sim_time
):
    """Print detalhado no terminal."""
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  BBTAN Bot — Snapshot Diagnostico")
    print(sep)

    # Config
    print(f"\n[CONFIG]")
    print(f"  GAME_REGION:   {cfg.GAME_REGION}")
    print(f"  BLOCK_SIZE:    {cfg.BLOCK_SIZE}px ({cfg.GRID_COLUMNS} colunas)")
    print(f"  BALL_RADIUS:   {cfg.BALL_RADIUS}px")
    print(
        f"  Paredes:       L={cfg.WALL_LEFT} R={cfg.WALL_RIGHT} T={cfg.WALL_TOP} B={cfg.WALL_BOTTOM}"
    )
    print(f"  Bolas:         {ball_count}")

    # Bola
    print(f"\n[BOLA]")
    if ball_pos:
        print(f"  Posicao:       ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")
    else:
        print(f"  Posicao:       NAO ENCONTRADA")

    # Blocos
    print(f"\n[BLOCOS] ({len(blocks)} detectados)")
    for i, b in enumerate(blocks):
        orient = (
            f" orient={b.tri_orientation.value}"
            if b.block_type == BlockType.TRIANGLE
            else ""
        )
        print(
            f"  #{i:2d}  {b.block_type.value:8s}  pos=({b.x:3d},{b.y:3d})  "
            f"size={b.w:3d}x{b.h:3d}  center=({b.cx:.0f},{b.cy:.0f})  hp={b.hp}{orient}"
        )

    # Power-ups
    print(f"\n[POWER-UPS] ({len(powerups)} detectados)")
    for i, p in enumerate(powerups):
        print(f"  #{i}  pos=({p.cx:.0f},{p.cy:.0f})  raio={p.radius:.0f}")

    # Simulacao
    print(f"\n[SIMULACAO]")
    print(
        f"  Angulos testados:  {len(all_results)} ({cfg.ANGLE_MIN}° a {cfg.ANGLE_MAX}°, step={cfg.ANGLE_STEP}°)"
    )
    print(f"  Tempo:             {sim_time:.3f}s")
    print(f"  Ball count:        {ball_count}x dano por colisao")

    # Top 10
    print(f"\n[TOP 10 ANGULOS]")
    print(
        f"  {'Rank':>4s}  {'Angulo':>7s}  {'Score':>8s}  {'Hits':>5s}  {'Destroy':>7s}  {'PwrUps':>6s}"
    )
    print(f"  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*6}")
    for i, r in enumerate(all_results[:10]):
        marker = " <<<" if i == 0 else ""
        print(
            f"  {i+1:4d}  {r.angle:6.1f}°  {r.score:8.0f}  {r.total_hits:5d}  "
            f"{r.blocks_destroyed:7d}  {r.powerups_collected:6d}{marker}"
        )

    # Piores 3 (contexto)
    print(f"\n[PIORES 3 ANGULOS]")
    for r in all_results[-3:]:
        print(f"  {r.angle:6.1f}°  score={r.score:.0f}  hits={r.total_hits}")

    # Melhor
    print(f"\n[DECISAO]")
    print(f"  Angulo escolhido:    {best.angle:.1f}°")
    print(f"  Score:               {best.score:.0f}")
    print(f"  Hits totais:         {best.total_hits}")
    print(f"  Blocos destruidos:   {best.blocks_destroyed}")
    print(f"  Power-ups coletados: {best.powerups_collected}")
    print(f"  Pontos na trajetoria:{len(best.path)}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="BBTAN Bot — Snapshot de 1 turno")
    parser.add_argument("--balls", type=int, default=1, help="Numero de bolas atual")
    parser.add_argument(
        "--save", action="store_true", help="Salva PNGs em vez de mostrar janelas"
    )
    args = parser.parse_args()

    ball_count = args.balls

    print("Capturando em 2s — posicione o jogo!")
    time.sleep(2)

    # ── 1. CAPTURA ──
    print("Capturando...")
    frame = capture_screen()
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    # ── 2. MASCARAS ──
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_blocks = cv2.inRange(hsv, cfg.BLOCK_HSV_LOW, cfg.BLOCK_HSV_HIGH)
    mask_powerups = (
        cv2.inRange(hsv, cfg.POWERUP_LASER_HSV_LOW, cfg.POWERUP_LASER_HSV_HIGH)
        | cv2.inRange(
            hsv, cfg.POWERUP_EXTRA_BALL_HSV_LOW, cfg.POWERUP_EXTRA_BALL_HSV_HIGH
        )
        | cv2.inRange(hsv, cfg.POWERUP_B_HSV_LOW, cfg.POWERUP_B_HSV_HIGH)
    )
    mask_ball = cv2.inRange(hsv, cfg.BALL_HSV_LOW, cfg.BALL_HSV_HIGH)

    # Combina as mascaras coloridas para visualizacao
    mask_vis = np.zeros_like(frame)
    mask_vis[mask_blocks > 0] = (0, 140, 255)  # laranja = blocos
    mask_vis[mask_powerups > 0] = (255, 255, 0)  # ciano = powerups
    mask_vis[mask_ball > 0] = (255, 255, 255)  # branco = bola

    # ── 3. DETECCAO ──
    blocks = detect_blocks(frame)
    powerups = detect_powerups(frame)
    ball_pos = detect_ball(frame)

    ball_x = ball_pos[0] if ball_pos else w / 2
    ball_y = ball_pos[1] if ball_pos else cfg.WALL_BOTTOM - 10

    # ── 4. SIMULACAO ──
    print(
        f"Simulando {cfg.ANGLE_MAX - cfg.ANGLE_MIN + 1} angulos com {ball_count} bola(s)..."
    )
    t0 = time.perf_counter()

    if blocks:
        best, all_results = find_best_angle(
            ball_x,
            ball_y,
            blocks,
            powerups,
            ball_count=ball_count,
        )
    else:
        best = SimResult(
            angle=90, score=0, total_hits=0, blocks_destroyed=0, powerups_collected=0
        )
        all_results = []

    sim_time = time.perf_counter() - t0

    # ── 5. MONTAR VISUALIZACOES ──

    # Painel 1: Frame cru + grid
    panel_raw = draw_grid(frame.copy())
    cv2.putText(
        panel_raw,
        "1. CAPTURA + GRID",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Painel 2: Mascaras HSV
    cv2.putText(
        mask_vis,
        "2. MASCARAS HSV",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        mask_vis,
        "laranja=blocos  ciano=powerup  branco=bola",
        (5, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (150, 150, 150),
        1,
    )

    # Painel 3: Deteccoes
    panel_detect = draw_detections(frame.copy(), blocks, powerups, ball_pos)
    cv2.putText(
        panel_detect,
        f"3. DETECCAO ({len(blocks)} blocos, {len(powerups)} pwrups)",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Painel 4: Heatmap de angulos + melhor trajetoria
    if blocks and all_results:
        panel_heat = draw_all_trajectories(
            frame.copy(),
            ball_x,
            ball_y,
            all_results,
            blocks,
            powerups,
            ball_count,
        )
        panel_heat = draw_best_trajectory(panel_heat, best, ball_x, ball_y)
    else:
        panel_heat = frame.copy()
    cv2.putText(
        panel_heat,
        f"4. SIMULACAO (melhor={best.angle:.0f}° score={best.score:.0f})",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Monta grid 2x2
    top_row = np.hstack([panel_raw, mask_vis])
    bot_row = np.hstack([panel_detect, panel_heat])
    combined = np.vstack([top_row, bot_row])

    # Adiciona barra de score
    final = draw_score_bar(combined, all_results)

    # ── 6. REPORT ──
    print_full_report(
        blocks, powerups, ball_pos, best, all_results, ball_count, sim_time
    )

    # ── 7. MOSTRAR / SALVAR ──
    if args.save:
        cv2.imwrite("snapshot_combined.png", final)
        cv2.imwrite("snapshot_raw.png", panel_raw)
        cv2.imwrite("snapshot_masks.png", mask_vis)
        cv2.imwrite("snapshot_detections.png", panel_detect)
        cv2.imwrite("snapshot_simulation.png", panel_heat)
        print(f"\nImagens salvas: snapshot_*.png")
    else:
        cv2.imshow("Snapshot: Analise completa", final)
        print(f"\nJanela aberta — pressione qualquer tecla para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
