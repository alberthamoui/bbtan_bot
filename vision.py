"""
vision.py — Captura de tela e detecção de objetos via OpenCV.

Responsabilidades:
  - Capturar a região do jogo (mss para performance)
  - Detectar blocos (quadrados e triângulos) com posição e tipo
  - Detectar power-ups circulares (+)
  - Localizar a bola branca na base
"""

import cv2
import numpy as np
import mss
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from pathlib import Path

import config as cfg

# ─────────────────────────────────────────────
# TIPOS DE DADOS
# ─────────────────────────────────────────────


class BlockType(Enum):
    SQUARE = "square"
    TRIANGLE = "triangle"
    UNKNOWN = "unknown"


class PowerUpType(Enum):
    EXTRA_BALL = "extra_ball"  # (+) ciano — bola extra
    LASER_H = "laser_h"       # (<>) azul — laser horizontal (ataca linha)
    LASER_V = "laser_v"       # (<>) azul — laser vertical (ataca coluna)
    BONUS_B = "bonus_b"       # (B) amarelo ouro — multiplicador de bônus
    UNKNOWN = "unknown"


class TriangleOrientation(Enum):
    """Orientação do triângulo baseada em onde fica o vértice agudo."""

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    UNKNOWN = "unknown"


@dataclass
class Block:
    """Representa um bloco detectado no jogo."""

    x: int  # canto superior esquerdo do bounding box
    y: int
    w: int  # largura
    h: int  # altura
    cx: float  # centro X
    cy: float  # centro Y
    block_type: BlockType = BlockType.UNKNOWN
    hp: int = 1  # hit points estimado (via OCR básico)
    vertices: np.ndarray = field(default_factory=lambda: np.array([]))
    tri_orientation: TriangleOrientation = TriangleOrientation.UNKNOWN
    row_index: int = -1   # 0 = mais baixa (perigosa), 6 = topo; -1 = desconhecido
    col_index: int = -1   # 0 = coluna mais à esquerda, 6 = mais à direita; -1 = desconhecido

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Retorna (x, y, w, h) para AABB collision."""
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)


@dataclass
class PowerUp:
    """Representa um power-up detectado no jogo."""

    cx: float
    cy: float
    radius: float
    power_up_type: PowerUpType = PowerUpType.UNKNOWN


@dataclass
class GameState:
    """Estado completo do jogo em um frame."""

    blocks: List[Block]
    powerups: List[PowerUp]
    ball_x: Optional[float]  # posição X da bola na base
    ball_y: Optional[float]  # posição Y da bola na base
    detected_ball_count: int  # bolas contadas visualmente na base
    game_over: bool = False  # tela de continue/game over detectada
    white_ratio: float = 0.0  # porcentagem de pixels brancos
    raw_frame: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    debug_frame: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


# ─────────────────────────────────────────────
# CAPTURA DE TELA
# ─────────────────────────────────────────────

_sct = mss.mss()  # instância global reutilizável

# ─────────────────────────────────────────────
# TEMPLATES DE DÍGITOS (carregados uma vez)
# ─────────────────────────────────────────────

_DIGIT_TEMPLATES: dict = {}  # {digit_str: binary_image}
_DIGIT_H = 28


def _load_digit_templates():
    if _DIGIT_TEMPLATES:
        return
    digits_dir = Path(__file__).resolve().parent / "templates" / "digits"
    if not digits_dir.exists():
        return
    for p in digits_dir.glob("*.png"):
        digit = p.stem
        if digit.isdigit():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _DIGIT_TEMPLATES[digit] = img


def _match_digit(crop: np.ndarray) -> Optional[str]:
    """Compara um crop binarizado contra todos os templates. Retorna o dígito mais parecido."""
    if not _DIGIT_TEMPLATES:
        return None
    h_norm = _DIGIT_H
    aspect = crop.shape[1] / max(crop.shape[0], 1)
    w_norm = max(1, round(h_norm * aspect))
    resized = cv2.resize(crop, (w_norm, h_norm), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    best_digit = None
    best_score = -1
    for digit, tmpl in _DIGIT_TEMPLATES.items():
        tw = tmpl.shape[1]
        target_w = max(tw, w_norm)
        pad_cand = np.zeros((h_norm, target_w), dtype=np.uint8)
        pad_tmpl = np.zeros((h_norm, target_w), dtype=np.uint8)
        ox_c = (target_w - w_norm) // 2
        pad_cand[:, ox_c:ox_c+w_norm] = binary
        ox_t = (target_w - tw) // 2
        pad_tmpl[:, ox_t:ox_t+tw] = tmpl
        match = np.sum(pad_cand == pad_tmpl)
        total = pad_cand.size
        score = match / total
        if score > best_score:
            best_score = score
            best_digit = digit
    if best_score < 0.6:
        return None
    return best_digit


def read_ball_count_template() -> int:
    """Lê o número de bolas via template matching na região BALL_COUNT_REGION."""
    _load_digit_templates()
    if not _DIGIT_TEMPLATES:
        return 0

    bx, by, bw, bh = cfg.BALL_COUNT_REGION
    mon = {"top": by, "left": bx, "width": bw, "height": bh}
    img = np.array(_sct.grab(mon), dtype=np.uint8)[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.BALL_COUNT_HSV_LOW, cfg.BALL_COUNT_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < 15 or h < 6:
            continue
        crop = mask[y:y+h, x:x+w]
        components.append((x, crop))

    components.sort(key=lambda c: c[0])

    if not components:
        return 0

    # Pula o primeiro componente (o "x" do prefixo)
    digit_components = components[1:]
    if not digit_components:
        return 0

    digits_str = ""
    for _, crop in digit_components:
        d = _match_digit(crop)
        if d is not None:
            digits_str += d

    if not digits_str:
        return 0
    return int(digits_str)


def capture_screen() -> np.ndarray:
    """Captura a região do jogo e retorna como array BGR (OpenCV)."""
    rx, ry, rw, rh = cfg.GAME_REGION
    monitor = {"top": ry, "left": rx, "width": rw, "height": rh}
    raw = _sct.grab(monitor)
    # mss retorna BGRA, converter para BGR
    frame = np.array(raw, dtype=np.uint8)[:, :, :3]
    return frame


# ─────────────────────────────────────────────
# DETECÇÃO DE BLOCOS
# ─────────────────────────────────────────────


def _classify_polygon(
    cnt: np.ndarray,
    approx: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[BlockType, TriangleOrientation]:
    """
    Usa a área do convex hull (não do contorno bruto) para classificar.
    O hull ignora buracos do texto HP dentro do bloco, evitando falsos triângulos.
      - hull_fill ~0.45-0.55  →  triângulo (ocupa metade da bbox)
      - hull_fill ~0.85-1.00  →  quadrado/bloco grande
    Threshold em 0.70 deixa margem confortável entre os dois grupos.
    """
    x, y, w, h = bbox
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    hull_fill = hull_area / max(w * h, 1)

    if hull_fill < 0.70:
        return BlockType.TRIANGLE, _get_triangle_orientation(approx, bbox)
    return BlockType.SQUARE, TriangleOrientation.UNKNOWN


def _get_triangle_orientation(
    approx: np.ndarray, bbox: Tuple[int, int, int, int]
) -> TriangleOrientation:
    """Determina a orientação do triângulo baseado na posição do vértice agudo."""
    x, y, w, h = bbox
    cx_box = x + w / 2
    cy_box = y + h / 2

    # Encontra o vértice mais distante do centro do bounding box
    pts = approx.reshape(-1, 2)
    distances = np.sqrt((pts[:, 0] - cx_box) ** 2 + (pts[:, 1] - cy_box) ** 2)
    tip_idx = np.argmax(distances)
    tip = pts[tip_idx]

    # Classifica baseado em qual quadrante do bbox o vértice está
    if tip[0] < cx_box and tip[1] < cy_box:
        return TriangleOrientation.TOP_LEFT
    elif tip[0] >= cx_box and tip[1] < cy_box:
        return TriangleOrientation.TOP_RIGHT
    elif tip[0] < cx_box and tip[1] >= cy_box:
        return TriangleOrientation.BOTTOM_LEFT
    else:
        return TriangleOrientation.BOTTOM_RIGHT


_HP_DIGIT_TEMPLATES: dict = {}
_HP_DIGIT_H = 20


def _load_hp_digit_templates():
    if _HP_DIGIT_TEMPLATES:
        return
    digits_dir = Path(__file__).resolve().parent / "templates" / "hp_digits"
    if not digits_dir.exists():
        return
    for p in digits_dir.glob("*.png"):
        digit = p.stem
        if digit.isdigit():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _HP_DIGIT_TEMPLATES[digit] = img


def _match_hp_digit(crop: np.ndarray) -> Optional[str]:
    if not _HP_DIGIT_TEMPLATES:
        return None
    h_norm = _HP_DIGIT_H
    aspect = crop.shape[1] / max(crop.shape[0], 1)
    w_norm = max(1, round(h_norm * aspect))
    resized = cv2.resize(crop, (w_norm, h_norm), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    best_digit = None
    best_score = -1
    for digit, tmpl in _HP_DIGIT_TEMPLATES.items():
        tw = tmpl.shape[1]
        target_w = max(tw, w_norm)
        pad_cand = np.zeros((h_norm, target_w), dtype=np.uint8)
        pad_tmpl = np.zeros((h_norm, target_w), dtype=np.uint8)
        ox_c = (target_w - w_norm) // 2
        pad_cand[:, ox_c:ox_c+w_norm] = binary
        ox_t = (target_w - tw) // 2
        pad_tmpl[:, ox_t:ox_t+tw] = tmpl
        match = np.sum(pad_cand == pad_tmpl)
        score = match / pad_cand.size
        if score > best_score:
            best_score = score
            best_digit = digit
    if best_score < 0.6:
        return None
    return best_digit


def _read_hp(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """
    Lê HP do bloco via template matching dos dígitos brancos.
    Fallback para estimativa por contagem de blobs se não houver templates.
    """
    pad_x = max(2, w // 6)
    pad_y = max(2, h // 6)
    x1, y1 = x + pad_x, y + pad_y
    x2, y2 = x + w - pad_x, y + h - pad_y

    if x2 <= x1 or y2 <= y1:
        return 1

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return 1

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    text_mask = cv2.inRange(hsv, cfg.HP_TEXT_HSV_LOW, cfg.HP_TEXT_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)

    if np.count_nonzero(text_mask) < 5:
        return 1

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)

    min_area = max(3, (x2 - x1) * (y2 - y1) // 100)

    # Tenta template matching primeiro
    _load_hp_digit_templates()
    if _HP_DIGIT_TEMPLATES:
        components = []
        for i in range(1, n_labels):
            cx, cy, cw, ch, ca = stats[i]
            if ca < min_area or ch < 4:
                continue
            crop = text_mask[cy:cy+ch, cx:cx+cw]
            components.append((cx, crop))

        if components:
            components.sort(key=lambda c: c[0])
            digits_str = ""
            for _, crop in components:
                d = _match_hp_digit(crop)
                if d is not None:
                    digits_str += d
            if digits_str:
                return int(digits_str)

    # Fallback: estimativa por contagem de blobs
    significant = sum(
        1 for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    )

    if significant <= 0:
        return 1
    elif significant == 1:
        return 5
    elif significant == 2:
        return 50
    elif significant == 3:
        return 500
    else:
        return 5000


def _get_row_index(cy: float) -> int:
    """
    Retorna o índice da fileira usando LEVEL_TOPS/LEVEL_BOTS.
    0 = mais baixa (perigosa), 6 = topo. Retorna -1 se fora de todas as faixas.
    """
    for i in range(len(cfg.LEVEL_TOPS)):
        if cfg.LEVEL_TOPS[i] <= cy <= cfg.LEVEL_BOTS[i]:
            return i
    return -1


def _get_col_index(cx: float) -> int:
    """
    Retorna o índice da coluna usando COL_LEFTS/COL_RIGHTS.
    0 = mais à esquerda, 6 = mais à direita. Retorna -1 se fora de todas.
    """
    for i in range(cfg.GRID_COLUMNS):
        if cfg.COL_LEFTS[i] <= cx <= cfg.COL_RIGHTS[i]:
            return i
    return -1


def detect_blocks(frame: np.ndarray) -> List[Block]:
    """Detecta todos os blocos coloridos no frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    raw_mask = cv2.inRange(hsv, cfg.BLOCK_HSV_LOW, cfg.BLOCK_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _MARGIN = 3

    blocks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.MIN_BLOCK_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Power-ups são pequenos demais para ser bloco
        if w < 35 or h < 35:
            continue

        # Contorno gigante (fusão morfológica de múltiplos blocos) — split por grade
        if area > cfg.MAX_BLOCK_AREA:
            for ri in range(len(cfg.LEVEL_TOPS)):
                cell_y = cfg.LEVEL_TOPS[ri]
                cell_y2 = cfg.LEVEL_BOTS[ri]
                if cell_y2 < y or cell_y > y + h:
                    continue
                for ci in range(cfg.GRID_COLUMNS):
                    cell_x = cfg.COL_LEFTS[ci]
                    cell_x2 = cfg.COL_RIGHTS[ci]
                    if cell_x2 < x or cell_x > x + w:
                        continue
                    cell_w = cell_x2 - cell_x
                    cell_h = cell_y2 - cell_y
                    cell_roi = mask[cell_y:cell_y2, cell_x:cell_x2]
                    if cell_roi.size == 0:
                        continue
                    fill = np.count_nonzero(cell_roi) / cell_roi.size
                    if fill < 0.40:
                        continue
                    raw_roi = raw_mask[cell_y:cell_y2, cell_x:cell_x2]
                    raw_cnts_cell, _ = cv2.findContours(
                        raw_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if raw_cnts_cell:
                        raw_cnt_cell = max(raw_cnts_cell, key=cv2.contourArea)
                        raw_peri = cv2.arcLength(raw_cnt_cell, True)
                        raw_approx = cv2.approxPolyDP(
                            raw_cnt_cell, cfg.POLY_APPROX_FACTOR * raw_peri, True
                        )
                        raw_approx[:, :, 0] += cell_x
                        raw_approx[:, :, 1] += cell_y
                        hull_area = cv2.contourArea(cv2.convexHull(raw_cnt_cell))
                        hull_fill = hull_area / max(cell_w * cell_h, 1)
                        if hull_fill < 0.60:
                            block_type = BlockType.TRIANGLE
                            tri_orient = _get_triangle_orientation(
                                raw_approx, (cell_x, cell_y, cell_w, cell_h))
                        else:
                            block_type = BlockType.SQUARE
                            tri_orient = TriangleOrientation.UNKNOWN
                        vertices = raw_approx.reshape(-1, 2)
                    else:
                        block_type = BlockType.SQUARE
                        tri_orient = TriangleOrientation.UNKNOWN
                        vertices = np.array([[cell_x, cell_y]])
                    cell_hp = _read_hp(frame, cell_x, cell_y, cell_w, cell_h)
                    blocks.append(Block(
                        x=cell_x, y=cell_y, w=cell_w, h=cell_h,
                        cx=cell_x + cell_w / 2.0, cy=cell_y + cell_h / 2.0,
                        block_type=block_type, hp=cell_hp,
                        vertices=vertices,
                        tri_orientation=tri_orient,
                        row_index=ri, col_index=ci,
                    ))
            continue

        c_left = _get_col_index(x + _MARGIN)
        c_right = _get_col_index(x + w - _MARGIN)
        r_top = _get_row_index(y + _MARGIN)
        r_bot = _get_row_index(y + h - _MARGIN)

        _MAX_OVERSHOOT = cfg.BLOCK_SIZE // 2
        if r_top == -1 and r_bot != -1:
            if y + _MARGIN >= cfg.LEVEL_TOPS[-1] - _MAX_OVERSHOOT:
                r_top = r_bot
        if r_bot == -1 and r_top != -1:
            if y + h - _MARGIN <= cfg.LEVEL_BOTS[0] + _MAX_OVERSHOOT:
                r_bot = r_top
        if c_left == -1 and c_right != -1:
            if x + _MARGIN >= cfg.COL_LEFTS[0] - _MAX_OVERSHOOT:
                c_left = c_right
        if c_right == -1 and c_left != -1:
            if x + w - _MARGIN <= cfg.COL_RIGHTS[-1] + _MAX_OVERSHOOT:
                c_right = c_left

        if -1 in (c_left, c_right, r_top, r_bot):
            continue

        cols_span = abs(c_right - c_left) + 1
        rows_span = abs(r_top - r_bot) + 1

        if cols_span == 1 and rows_span == 1:
            # Bloco 1×1 — classifica usando contorno da raw_mask (sem
            # morfologia que infla triângulos para parecerem quadrados)
            raw_roi = raw_mask[y:y+h, x:x+w]
            raw_cnts, _ = cv2.findContours(
                raw_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if raw_cnts:
                raw_cnt = max(raw_cnts, key=cv2.contourArea)
                raw_peri = cv2.arcLength(raw_cnt, True)
                raw_approx = cv2.approxPolyDP(
                    raw_cnt, cfg.POLY_APPROX_FACTOR * raw_peri, True
                )
                raw_approx[:, :, 0] += x
                raw_approx[:, :, 1] += y

                hull_area = cv2.contourArea(cv2.convexHull(raw_cnt))
                hull_fill = hull_area / max(w * h, 1)

                if hull_fill < 0.60:
                    block_type = BlockType.TRIANGLE
                    tri_orient = _get_triangle_orientation(raw_approx, (x, y, w, h))
                else:
                    block_type = BlockType.SQUARE
                    tri_orient = TriangleOrientation.UNKNOWN

                vertices = raw_approx.reshape(-1, 2)
            else:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, cfg.POLY_APPROX_FACTOR * peri, True)
                block_type, tri_orient = _classify_polygon(cnt, approx, (x, y, w, h))
                vertices = approx.reshape(-1, 2)

            hp = _read_hp(frame, x, y, w, h)
            blocks.append(Block(
                x=x, y=y, w=w, h=h,
                cx=x + w / 2.0, cy=y + h / 2.0,
                block_type=block_type, hp=hp,
                vertices=vertices,
                tri_orientation=tri_orient,
                row_index=r_top, col_index=c_left,
            ))

        elif cols_span == 2 and rows_span == 2:
            # Candidato a bloco grande 2×2 — validar que as 4 cells estão preenchidas
            # com a mesma cor (bloco real vs 3 blocos fundidos por morfologia)
            r_min_2 = min(r_top, r_bot)
            c_min_2 = min(c_left, c_right)
            is_real_2x2 = True
            cell_fills = []
            for ri in range(r_min_2, r_min_2 + 2):
                for ci in range(c_min_2, c_min_2 + 2):
                    cy1 = max(0, cfg.LEVEL_TOPS[ri])
                    cy2 = min(mask.shape[0], cfg.LEVEL_BOTS[ri])
                    cx1 = max(0, cfg.COL_LEFTS[ci])
                    cx2 = min(mask.shape[1], cfg.COL_RIGHTS[ci])
                    cell_roi = mask[cy1:cy2, cx1:cx2]
                    if cell_roi.size == 0:
                        is_real_2x2 = False
                        break
                    fill = np.count_nonzero(cell_roi) / cell_roi.size
                    cell_fills.append(fill)
                    if fill < 0.50:
                        is_real_2x2 = False
                        break
                if not is_real_2x2:
                    break

            # Verifica cor uniforme: uma única color range deve cobrir >50% de toda a área
            if is_real_2x2:
                bx1 = max(0, cfg.COL_LEFTS[c_min_2])
                bx2 = min(hsv.shape[1], cfg.COL_RIGHTS[c_min_2 + 1])
                by1 = max(0, cfg.LEVEL_TOPS[r_min_2 + 1])
                by2 = min(hsv.shape[0], cfg.LEVEL_BOTS[r_min_2])
                block_hsv = hsv[by1:by2, bx1:bx2]
                block_area = block_hsv.shape[0] * block_hsv.shape[1]
                color_ok = False
                for _name, low, high in cfg.BLOCK_COLOR_RANGES:
                    color_mask = cv2.inRange(block_hsv, low, high)
                    if np.count_nonzero(color_mask) > block_area * 0.40:
                        color_ok = True
                        break
                if not color_ok:
                    is_real_2x2 = False

            if is_real_2x2:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, cfg.POLY_APPROX_FACTOR * peri, True)
                hp = _read_hp(frame, x, y, w, h)
                blocks.append(Block(
                    x=x, y=y, w=w, h=h,
                    cx=x + w / 2.0, cy=y + h / 2.0,
                    block_type=BlockType.SQUARE, hp=hp,
                    vertices=approx.reshape(-1, 2),
                    tri_orientation=TriangleOrientation.UNKNOWN,
                    row_index=r_min_2, col_index=c_min_2,
                ))
            else:
                # Não é 2x2 real — split em blocos individuais
                r_min = min(r_top, r_bot)
                r_max = max(r_top, r_bot)
                c_min = min(c_left, c_right)
                c_max = max(c_left, c_right)
                for ri in range(r_min, r_max + 1):
                    for ci in range(c_min, c_max + 1):
                        cell_x = cfg.COL_LEFTS[ci]
                        cell_w = cfg.COL_RIGHTS[ci] - cell_x
                        cell_y = cfg.LEVEL_TOPS[ri]
                        cell_h = cfg.LEVEL_BOTS[ri] - cell_y
                        cy1 = max(0, cell_y)
                        cy2 = min(mask.shape[0], cell_y + cell_h)
                        cx1 = max(0, cell_x)
                        cx2 = min(mask.shape[1], cell_x + cell_w)
                        cell_roi = mask[cy1:cy2, cx1:cx2]
                        if cell_roi.size == 0 or np.count_nonzero(cell_roi) < cell_w * cell_h * 0.15:
                            continue
                        cell_hp = _read_hp(frame, cell_x, cell_y, cell_w, cell_h)
                        blocks.append(Block(
                            x=cell_x, y=cell_y, w=cell_w, h=cell_h,
                            cx=cell_x + cell_w / 2.0, cy=cell_y + cell_h / 2.0,
                            block_type=BlockType.SQUARE, hp=cell_hp,
                            vertices=np.array([[cell_x, cell_y]]),
                            tri_orientation=TriangleOrientation.UNKNOWN,
                            row_index=ri, col_index=ci,
                        ))

        else:
            # Contornos fundidos (2×1, 1×2, etc.) — separa em blocos individuais
            r_min = min(r_top, r_bot)
            r_max = max(r_top, r_bot)
            c_min = min(c_left, c_right)
            c_max = max(c_left, c_right)
            for ri in range(r_min, r_max + 1):
                for ci in range(c_min, c_max + 1):
                    cell_x = cfg.COL_LEFTS[ci]
                    cell_w = cfg.COL_RIGHTS[ci] - cell_x
                    cell_y = cfg.LEVEL_TOPS[ri]
                    cell_h = cfg.LEVEL_BOTS[ri] - cell_y
                    # Verifica se a célula realmente tem conteúdo de bloco
                    cy1 = max(0, cell_y)
                    cy2 = min(mask.shape[0], cell_y + cell_h)
                    cx1 = max(0, cell_x)
                    cx2 = min(mask.shape[1], cell_x + cell_w)
                    cell_roi = mask[cy1:cy2, cx1:cx2]
                    if cell_roi.size == 0 or np.count_nonzero(cell_roi) < cell_w * cell_h * 0.15:
                        continue
                    cell_hp = _read_hp(frame, cell_x, cell_y, cell_w, cell_h)
                    blocks.append(Block(
                        x=cell_x, y=cell_y, w=cell_w, h=cell_h,
                        cx=cell_x + cell_w / 2.0, cy=cell_y + cell_h / 2.0,
                        block_type=BlockType.SQUARE, hp=cell_hp,
                        vertices=np.array([[cell_x, cell_y]]),
                        tri_orientation=TriangleOrientation.UNKNOWN,
                        row_index=ri, col_index=ci,
                    ))

    return blocks


# ─────────────────────────────────────────────
# DETECÇÃO DE POWER-UPS
# ─────────────────────────────────────────────


_POWERUP_HSV_RANGES = [
    (PowerUpType.EXTRA_BALL, cfg.POWERUP_EXTRA_BALL_HSV_LOW, cfg.POWERUP_EXTRA_BALL_HSV_HIGH),
    ("LASER",                cfg.POWERUP_LASER_HSV_LOW,      cfg.POWERUP_LASER_HSV_HIGH),
    (PowerUpType.BONUS_B,    cfg.POWERUP_B_HSV_LOW,          cfg.POWERUP_B_HSV_HIGH),
]


def _detect_laser_orientation(frame_gray: np.ndarray, cx: float, cy: float, radius: float) -> PowerUpType:
    """
    Determina se o laser é horizontal ou vertical analisando o padrão
    interno do ícone (setas <> vs setas verticais).
    """
    r = int(radius)
    h, w = frame_gray.shape[:2]
    x1 = max(0, int(cx) - r)
    y1 = max(0, int(cy) - r)
    x2 = min(w, int(cx) + r)
    y2 = min(h, int(cy) + r)

    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return PowerUpType.LASER_H

    h_proj = np.var(np.mean(roi, axis=1))
    v_proj = np.var(np.mean(roi, axis=0))

    return PowerUpType.LASER_V if h_proj > v_proj else PowerUpType.LASER_H


def detect_powerups(frame: np.ndarray) -> List[PowerUp]:
    """Detecta todos os tipos de power-up circulares no frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    powerups = []
    for ptype, low, high in _POWERUP_HSV_RANGES:
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity > 0.6:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                if radius < 7 or radius > 15:
                    continue
                if ptype == "LASER":
                    final_type = _detect_laser_orientation(gray, cx, cy, radius)
                else:
                    final_type = ptype
                powerups.append(PowerUp(cx=cx, cy=cy, radius=radius, power_up_type=final_type))

    return powerups


# ─────────────────────────────────────────────
# DETECÇÃO DA BOLA
# ─────────────────────────────────────────────


def detect_ball(frame: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Localiza a bola branca na metade inferior do frame.
    Retorna (cx, cy) ou None se não encontrar.
    """
    h, w = frame.shape[:2]
    # Só procura na metade inferior (onde a bola fica)
    roi = frame[h // 2 :, :]
    y_offset = h // 2

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.BALL_HSV_LOW, cfg.BALL_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circularity = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 500:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)

        if circularity > best_circularity and circularity > 0.5:
            best_circularity = circularity
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"] + y_offset
                best = (cx, cy)

    return best



def count_balls(frame: np.ndarray) -> int:
    """
    Conta visualmente quantas bolas brancas estão na base da tela.
    No BBTAN, entre turnos todas as bolas ficam empilhadas/agrupadas
    na parte inferior. Conta o número de blobs brancos separados
    ou estima pela área total de pixels brancos na região da base.
    """
    h, w = frame.shape[:2]

    # Região da base: últimos 15% da tela (onde as bolas ficam entre turnos)
    base_top = int(h * 0.80)
    roi = frame[base_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.BALL_HSV_LOW, cfg.BALL_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Método 1: contar contornos individuais com tamanho de bola
    # Cada bola tem área ~pi*r^2
    ball_area = np.pi * cfg.BALL_RADIUS * cfg.BALL_RADIUS
    min_ball_area = ball_area * 0.3  # margem pra bolas parcialmente visíveis
    max_single_ball = ball_area * 2.5  # uma bola sozinha

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_ball_area:
            continue

        if area <= max_single_ball:
            # Uma bola individual
            count += 1
        else:
            # Grupo de bolas sobrepostas: estima quantas tem pela área
            count += max(1, round(area / ball_area))

    # Método 2 (fallback): área total de pixels brancos / área de 1 bola
    if count == 0:
        total_white = np.count_nonzero(mask)
        if total_white > min_ball_area:
            count = max(1, round(total_white / ball_area))

    return max(1, count)  # mínimo 1 bola


# ─────────────────────────────────────────────
# DETECÇÃO DE FIM DE RODADA
# ─────────────────────────────────────────────


def detect_round_end() -> bool:
    """
    Verifica se pelo menos 2 das 4 regiões indicadoras contêm a cor rosa.
    """
    hits = 0
    for (rx, ry, rw, rh) in cfg.ROUND_END_REGIONS:
        mon = {"top": ry, "left": rx, "width": rw, "height": rh}
        img = np.array(_sct.grab(mon), dtype=np.uint8)[:, :, :3]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cfg.ROUND_END_HSV_LOW, cfg.ROUND_END_HSV_HIGH)
        if np.count_nonzero(mask) > mask.size * 0.1:
            hits += 1
    return hits >= 2


# ─────────────────────────────────────────────
# DETECÇÃO DE GAME OVER (tela branca)
# ─────────────────────────────────────────────

# Porcentagem mínima de pixels brancos para considerar game over
GAMEOVER_WHITE_RATIO = 0.25  # 25% da tela branca = popup aberto


def detect_game_over(frame: np.ndarray) -> tuple:
    """
    Detecta se a tela de game over/continue apareceu.
    A tela de continue tem uma área branca grande no centro.

    Retorna (is_game_over: bool, white_ratio: float)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pixels brancos: valor > 200
    white_mask = gray > 200
    white_ratio = np.count_nonzero(white_mask) / gray.size
    return white_ratio >= GAMEOVER_WHITE_RATIO, white_ratio


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────


def scan_game() -> GameState:
    """Executa o pipeline completo de visão e retorna o estado do jogo."""
    frame = capture_screen()

    # Checa game over ANTES de detectar blocos (evita processamento inútil)
    is_game_over, white_ratio = detect_game_over(frame)
    if is_game_over:
        return GameState(
            blocks=[],
            powerups=[],
            ball_x=None,
            ball_y=None,
            detected_ball_count=0, 
            game_over=True,
            white_ratio=white_ratio,
            raw_frame=frame,
            debug_frame=frame.copy(),
        )

    blocks = detect_blocks(frame)
    powerups = detect_powerups(frame)
    ball_pos = detect_ball(frame)
    n_balls = read_ball_count_template()
    if n_balls == 0:
        n_balls = count_balls(frame)

    ball_x = ball_pos[0] if ball_pos else None
    ball_y = ball_pos[1] if ball_pos else None

    # Frame de debug com anotações
    debug = frame.copy()
    for b in blocks:
        is_large = b.w > cfg.BLOCK_SIZE * 1.5
        if b.block_type == BlockType.TRIANGLE:
            color = (0, 0, 255)     # vermelho
            tipo = "TRI"
        elif is_large:
            color = (0, 255, 0)     # verde
            tipo = "GRANDE"
        else:
            color = (255, 0, 0)     # azul
            tipo = "BLK"
        cv2.rectangle(debug, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        label = f"{tipo} hp={b.hp} L{7 - b.row_index}C{b.col_index}"
        cx_label = b.x + b.w // 2
        cy_label = b.y + b.h // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(debug, label, (cx_label - tw // 2, cy_label + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(debug, label, (cx_label - tw // 2, cy_label + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    _POWERUP_DEBUG_LABEL = {
        PowerUpType.EXTRA_BALL: "+",
        PowerUpType.LASER_H:    "<>",
        PowerUpType.LASER_V:    "^v",
        PowerUpType.BONUS_B:    "B",
        PowerUpType.UNKNOWN:    "?",
    }
    for p in powerups:
        cv2.circle(debug, (int(p.cx), int(p.cy)), int(p.radius), (255, 255, 0), 2)
        cv2.putText(
            debug,
            _POWERUP_DEBUG_LABEL[p.power_up_type],
            (int(p.cx) - 5, int(p.cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

    if ball_pos:
        cv2.circle(debug, (int(ball_x), int(ball_y)), 8, (0, 0, 255), 2)

    return GameState(
        blocks=blocks,
        powerups=powerups,
        ball_x=ball_x,
        ball_y=ball_y,
        detected_ball_count=n_balls,
        game_over=False,
        white_ratio=white_ratio,
        raw_frame=frame,
        debug_frame=debug,
    )


# ─────────────────────────────────────────────
# UTILITÁRIO DE DEBUG
# ─────────────────────────────────────────────


def show_debug(state: GameState, wait: int = 0):
    """Mostra o frame de debug com anotações. Pressione qualquer tecla para fechar."""
    cv2.imshow("BBTAN Bot - Debug", state.debug_frame)
    cv2.waitKey(wait)


if __name__ == "__main__":
    # Teste rápido de detecção
    print("Capturando tela...")
    state = scan_game()
    print(f"Blocos: {len(state.blocks)}")
    for b in state.blocks:
        print(f"  {b.block_type.value} @ ({b.x},{b.y}) {b.w}x{b.h} hp={b.hp}")
    print(f"Power-ups: {len(state.powerups)}")
    print(f"Bola: ({state.ball_x}, {state.ball_y})")
    show_debug(state, wait=0)
    cv2.destroyAllWindows()
