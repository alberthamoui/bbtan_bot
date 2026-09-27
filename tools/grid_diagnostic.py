"""
grid_diagnostic.py — Visualizacao de diagnostico da grade e blocos detectados.

Captura 1 frame, desenha por cima:
  1. Borda da area jogavel (WALL_*)
  2. Overlay completo da grade (todas as celulas de COL_LEFTS/RIGHTS x LEVEL_TOPS/BOTS)
  3. Label R{row_index}C{col_index} no centro de cada celula
  4. Blocos detectados por detect_blocks() com bbox real e label "hp=X L{7-ri}C{ci}"
  5. Fileira 0 (row_index=0, mais baixa/perigosa) destacada em cor distinta

Salva em tools/grid_diagnostic.png e abre janela cv2.
Objetivo: confirmar visualmente se a fileira 0 esta coberta pela grade
e se detect_blocks() encontra os blocos que estao la.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from vision import capture_screen, detect_blocks

OUT_PATH = Path(__file__).resolve().parent / "grid_diagnostic.png"
WINDOW = "Diagnostico: Grade da tela"

# Cores (BGR)
COLOR_WALL = (0, 255, 0)  # verde brilhante — borda WALL_*
COLOR_GRID = (90, 90, 90)  # cinza — linhas de grade
COLOR_CELL_LABEL = (200, 200, 200)  # cinza claro — R{r}C{c}
COLOR_BLOCK_BBOX = (0, 255, 255)  # ciano — contorno detectado
COLOR_BLOCK_LABEL_BG = (0, 0, 0)  # sombra preta
COLOR_BLOCK_LABEL = (0, 255, 255)  # ciano
COLOR_ROW0_FILL = (0, 180, 255)  # laranja — destaque da fileira 0
ALPHA_ROW0 = 0.25


def _put_text(img, text, org, color, scale=0.4, thickness=1, shadow=True):
    if shadow:
        cv2.putText(
            img,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def _draw_row0_highlight(img):
    """Preenche as celulas da fileira row_index=0 (mais baixa) em cor distinta."""
    ri = 0
    y_top = cfg.LEVEL_TOPS[ri]
    y_bot = cfg.LEVEL_BOTS[ri]
    overlay = img.copy()
    for ci in range(cfg.GRID_COLUMNS):
        x_left = cfg.COL_LEFTS[ci]
        x_right = cfg.COL_RIGHTS[ci]
        cv2.rectangle(
            overlay, (x_left, y_top), (x_right, y_bot), COLOR_ROW0_FILL, thickness=-1
        )
    cv2.addWeighted(overlay, ALPHA_ROW0, img, 1 - ALPHA_ROW0, 0, dst=img)


def _draw_grid(img):
    """Todas as celulas: retangulo fino + label R{ri}C{ci} no centro."""
    for ri in range(len(cfg.LEVEL_TOPS)):
        y_top = cfg.LEVEL_TOPS[ri]
        y_bot = cfg.LEVEL_BOTS[ri]
        for ci in range(cfg.GRID_COLUMNS):
            x_left = cfg.COL_LEFTS[ci]
            x_right = cfg.COL_RIGHTS[ci]
            cv2.rectangle(img, (x_left, y_top), (x_right, y_bot), COLOR_GRID, 1)
            label = f"R{ri}C{ci}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            tx = (x_left + x_right) // 2 - tw // 2
            ty = (y_top + y_bot) // 2 + th // 2
            _put_text(img, label, (tx, ty), COLOR_CELL_LABEL, scale=0.35, thickness=1)


def _draw_wall_border(img):
    """Retangulo grosso mostrando WALL_LEFT/RIGHT/TOP/BOTTOM."""
    cv2.rectangle(
        img,
        (cfg.WALL_LEFT, cfg.WALL_TOP),
        (cfg.WALL_RIGHT, cfg.WALL_BOTTOM),
        COLOR_WALL,
        2,
    )
    _put_text(
        img,
        f"WALL L={cfg.WALL_LEFT} R={cfg.WALL_RIGHT} "
        f"T={cfg.WALL_TOP} B={cfg.WALL_BOTTOM}",
        (cfg.WALL_LEFT + 4, cfg.WALL_TOP + 14),
        COLOR_WALL,
        scale=0.4,
        thickness=1,
    )


def _draw_blocks(img, blocks):
    """Bbox real de cada Block detectado + label hp/L/C."""
    for b in blocks:
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), COLOR_BLOCK_BBOX, 2)
        label = f"hp={b.hp} L{7 - b.row_index}C{b.col_index}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Encaixa acima do bloco quando possivel, se nao encaixa dentro
        ty = b.y - 4
        if ty - th < 0:
            ty = b.y + th + 4
        tx = b.x
        _put_text(img, label, (tx, ty), COLOR_BLOCK_LABEL, scale=0.4, thickness=1)


def _print_summary(blocks):
    print("=" * 60)
    print(f"  GAME_REGION: {cfg.GAME_REGION}")
    print(
        f"  WALLS: L={cfg.WALL_LEFT} R={cfg.WALL_RIGHT} "
        f"T={cfg.WALL_TOP} B={cfg.WALL_BOTTOM}"
    )
    print(f"  BLOCK_SIZE: {cfg.BLOCK_SIZE}  GRID_COLUMNS: {cfg.GRID_COLUMNS}")
    print(f"  LEVEL_TOPS: {cfg.LEVEL_TOPS}")
    print(f"  LEVEL_BOTS: {cfg.LEVEL_BOTS}")
    print(f"  COL_LEFTS : {cfg.COL_LEFTS}")
    print(f"  COL_RIGHTS: {cfg.COL_RIGHTS}")
    print("-" * 60)
    print(f"  Detectados: {len(blocks)} blocos")
    row0 = [b for b in blocks if b.row_index == 0]
    print(f"  Blocos com row_index=0 (fileira mais baixa): {len(row0)}")
    for b in row0:
        print(
            f"    L7-{b.row_index}=L{7 - b.row_index}C{b.col_index}"
            f"  bbox=({b.x},{b.y},{b.w}x{b.h})  hp={b.hp}"
            f"  type={b.block_type.value}"
        )
    print("=" * 60)


def main():
    print("[grid_diagnostic] Capturando frame...")
    frame = capture_screen()
    print(f"[grid_diagnostic] Frame shape: {frame.shape}")

    print("[grid_diagnostic] Rodando detect_blocks()...")
    blocks = detect_blocks(frame)

    canvas = frame.copy()
    _draw_row0_highlight(canvas)
    _draw_grid(canvas)
    _draw_wall_border(canvas)
    _draw_blocks(canvas, blocks)

    _print_summary(blocks)

    cv2.imwrite(str(OUT_PATH), canvas)
    print(f"[grid_diagnostic] Salvo em: {OUT_PATH}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW, canvas)
    print("[grid_diagnostic] Pressione qualquer tecla na janela para sair.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
