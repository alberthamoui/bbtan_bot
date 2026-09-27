"""
desenhar_caminho.py — Tira um print da tela do jogo AGORA e desenha o caminho ideal.

Roda a mesma visão + simulação do bot (scan_game + find_best_angle) sobre a
GAME_REGION no momento em que é executado e salva em prints/ a imagem com:
  - blocos e power-ups detectados (contorno fino)
  - trajetória da 1a bola no ângulo escolhido (laranja) + colisões (vermelho)
  - ângulo, nº de bolas, score, hits e blocos destruídos no topo

O nº de bolas vem do contador da tela ("x14"), lido pelo mesmo OCR do bot.
Use --balls pra forçar outro valor.

Uso:
  python desenhar_caminho.py               -> print + caminho, na hora
  python desenhar_caminho.py --delay 3     -> espera 3s antes (pra focar o jogo)
  python desenhar_caminho.py --balls 12    -> simula com 12 bolas
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config as cfg
from vision import BlockType, scan_game
from physics import find_best_angle

SAIDA = Path(__file__).resolve().parent / "prints"

COR_CAMINHO = (0, 200, 255)  # laranja
COR_COLISAO = (0, 0, 255)  # vermelho
COR_BLOCO = (255, 160, 0)  # azul claro
COR_TRI = (180, 0, 255)  # rosa
COR_POWERUP = (255, 255, 0)  # ciano
COR_BOLA = (255, 255, 255)


def _texto(img, texto, org, cor, escala=0.5):
    """Texto com contorno preto. Só ASCII: o putText não renderiza acento."""
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Tira um print do jogo e desenha o caminho ideal"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Segundos de espera antes do print"
    )
    parser.add_argument(
        "--balls",
        type=int,
        default=0,
        help="Forca o numero de bolas (0 = le o contador da tela)",
    )
    args = parser.parse_args()

    if args.delay > 0:
        print(f"Print em {args.delay:.0f}s...")
        time.sleep(args.delay)

    state = scan_game()
    if state.game_over:
        print(f"Tela de game over ({state.white_ratio:.0%} branca) — nada a desenhar.")
        return
    if state.ball_x is None:
        print("Bola nao encontrada — a rodada ainda esta acontecendo?")
        return
    if not state.blocks:
        print("Nenhum bloco detectado.")
        return

    balls = args.balls if args.balls > 0 else max(1, state.detected_ball_count)
    bx, by = state.ball_x, state.ball_y

    t0 = time.perf_counter()
    best, _ = find_best_angle(bx, by, state.blocks, state.powerups, ball_count=balls)
    sim = time.perf_counter() - t0

    vis = state.raw_frame.copy()

    # Detecções (fino, pra conferir o que a simulação enxergou)
    for b in state.blocks:
        if b.block_type == BlockType.TRIANGLE and len(b.vertices) >= 3:
            pts = b.vertices.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis, [pts], True, COR_TRI, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), COR_BLOCO, 1)
    for p in state.powerups:
        cv2.circle(vis, (int(p.cx), int(p.cy)), int(p.radius), COR_POWERUP, 1)

    # Caminho ideal
    if len(best.path) >= 2:
        pts = np.array([(bx, by)] + best.path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, COR_CAMINHO, 2, cv2.LINE_AA)
    for cx, cy in best.collisions:
        cv2.circle(vis, (int(cx), int(cy)), 4, COR_COLISAO, -1)
    cv2.circle(vis, (int(bx), int(by)), cfg.BALL_RADIUS, COR_BOLA, 2)

    _texto(vis, f"angulo {best.angle:.0f} graus  |  {balls} bola(s)", (6, 18), COR_BOLA)
    _texto(
        vis,
        f"score {best.score:.0f}  hits {best.total_hits}  destroi {best.blocks_destroyed}",
        (6, 38),
        COR_CAMINHO,
        escala=0.45,
    )

    SAIDA.mkdir(exist_ok=True)
    arquivo = SAIDA / f"{datetime.now():%Y%m%d_%H%M%S}.png"
    if not cv2.imwrite(str(arquivo), vis):
        print(f"[!] Falha ao salvar {arquivo}")
        return

    print(
        f"Angulo {best.angle:.0f}  |  {balls} bola(s)  |  score {best.score:.0f}  "
        f"hits {best.total_hits}  destroi {best.blocks_destroyed}  ({sim:.1f}s)"
    )
    print(f"Salvo em prints/{arquivo.name}")


if __name__ == "__main__":
    main()
