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
    hp_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8), repr=False
    )  # text_mask gerada por _get_block_text_mask para debug DEBUG_MASK_HP

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


def _read_round_number() -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Lê o número da rodada via template matching na região fixa da tela.
    Retorna (round_number, mask_debug).
    """
    _load_digit_templates()
    if not _DIGIT_TEMPLATES:
        return None, None

    rx, ry, rw, rh = 1527, 191, 101, 50
    mon = {"top": ry, "left": rx, "width": rw, "height": rh}
    img = np.array(_sct.grab(mon), dtype=np.uint8)[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    round_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([179, 30, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    round_mask = cv2.morphologyEx(round_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(round_mask, connectivity=8)

    components = []
    for i in range(1, n_labels):
        sx, sy, sw, sh, sa = stats[i]
        if sa < 15 or sh < 6:
            continue
        crop = round_mask[sy:sy+sh, sx:sx+sw]
        components.append((sx, crop))

    if not components:
        return None, round_mask

    components.sort(key=lambda c: c[0])

    digits_str = ""
    for _, crop in components:
        d = _match_digit(crop)
        if d is not None:
            digits_str += d

    if not digits_str:
        return None, round_mask
    return int(digits_str), round_mask


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
        return BlockType.TRIANGLE, TriangleOrientation.UNKNOWN
    return BlockType.SQUARE, TriangleOrientation.UNKNOWN


def _get_triangle_orientation(raw_roi: np.ndarray) -> TriangleOrientation:
    """Determina a orientação do triângulo via centroide do mask na célula."""
    M = cv2.moments(raw_roi)
    if M["m00"] < 1:
        return TriangleOrientation.UNKNOWN
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center_x = raw_roi.shape[1] / 2.0
    center_y = raw_roi.shape[0] / 2.0
    dx = cx - center_x
    dy = cy - center_y
    if abs(dx) < 5 and abs(dy) < 5:
        return TriangleOrientation.UNKNOWN
    if dx > 0 and dy > 0:
        return TriangleOrientation.BOTTOM_RIGHT
    elif dx > 0 and dy < 0:
        return TriangleOrientation.TOP_RIGHT
    elif dx < 0 and dy > 0:
        return TriangleOrientation.BOTTOM_LEFT
    else:
        return TriangleOrientation.TOP_LEFT


_HP_DIGIT_TEMPLATES: dict = {}
_HP_DIGIT_H = 20


def _load_hp_digit_templates():
    if _HP_DIGIT_TEMPLATES:
        return
    # antigo: templates/digits (28px, fonte do contador de bolas) — agora le
    # templates/hp_digits/<digito>.png (20px, consolidados via
    # consolidate_hp_templates.py a partir das amostras <digito>_<timestamp>.png).
    # Filtra apenas arquivos com nome de 1 caractere, ignorando as amostras brutas.
    digits_dir = Path(__file__).resolve().parent / "templates" / "hp_digits"
    if not digits_dir.exists():
        return
    for p in digits_dir.glob("*.png"):
        digit = p.stem
        if len(digit) != 1 or not digit.isdigit():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        src_h, src_w = img.shape[:2]
        if src_h != _HP_DIGIT_H:
            aspect = src_w / max(src_h, 1)
            new_w = max(1, round(_HP_DIGIT_H * aspect))
            img = cv2.resize(img, (new_w, _HP_DIGIT_H), interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        _HP_DIGIT_TEMPLATES[digit] = binary


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


def _get_block_text_mask(roi_bgr: np.ndarray, is_triangle: bool = False) -> np.ndarray:
    """
    Isola o texto (número de HP) dentro do bloco.

    Texto é sempre quase preto (V baixo). Pra evitar pegar pixels escuros
    do background, restringe a busca ao convex hull do bloco — assim o
    texto, que fica dentro da área colorida (centro do quadrado), é
    capturado mas o background fica de fora.

    Para triângulos, o hull exato corta o texto que toca a hipotenusa.
    Quando is_triangle=True, usa a ROI inteira como block_region (a ROI
    já está cropada pra dentro da célula, então não pega blocos vizinhos).

    Retorna mask binária 0/255 com o texto em branco e o resto em preto.
    """
    if roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    total = hsv.shape[0] * hsv.shape[1]

    best_mask = None
    best_coverage = 0
    for _name, low, high in cfg.BLOCK_COLOR_RANGES:
        m = cv2.inRange(hsv, low, high)
        coverage = int(np.count_nonzero(m))
        if coverage > best_coverage:
            best_coverage = coverage
            best_mask = m

    if best_mask is None or best_coverage < total * 0.15:
        # Cor do bloco não identificada — fallback ao threshold escuro
        text_mask = cv2.inRange(hsv, cfg.HP_TEXT_HSV_LOW, cfg.HP_TEXT_HSV_HIGH)
    else:
        if is_triangle:
            # Triângulo: ROI inteira como block_region. O hull do colorido
            # cortava o dígito que toca a hipotenusa; a ROI já está cropada
            # pra dentro da célula da grade, então não pega blocos vizinhos.
            block_region = np.full_like(best_mask, 255)
        else:
            # Quadrado: hull cobre toda a área colorida + os buracos do texto
            block_region = np.zeros_like(best_mask)
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest)
                cv2.fillPoly(block_region, [hull], 255)
            else:
                block_region = best_mask

        # Pega o numero dentro dos blocos
        # Texto é sempre quase preto — procura pixels escuros dentro do bloco
        # antigo: cv2.inRange(v_channel, 0, 90)
        dark = cv2.inRange(v_channel, 0, 130)
        text_mask = cv2.bitwise_and(dark, block_region)

        if is_triangle and cfg.VISION_VERBOSE:
            print(
                f"[TRI hp_mask] block_region={int(np.count_nonzero(block_region))}"
                f"  dark={int(np.count_nonzero(dark))}"
                f"  text_mask={int(np.count_nonzero(text_mask))}"
                f"  roi={roi_bgr.shape[1]}x{roi_bgr.shape[0]}"
            )

    # Limpa ruído pequeno
    # antigo: cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)
    return text_mask


def _read_hp(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int,
             is_triangle: bool = False,
             is_2x2: bool = False) -> Tuple[int, np.ndarray]:
    """
    Lê HP do bloco via template matching contra templates/hp_digits/.
    Templates esperam fundo preto + dígito branco; _get_block_text_mask
    produz essa orientação.

    is_triangle=True propaga pro _get_block_text_mask que dilata o hull
    pra recuperar pixels de texto cortados pela hipotenusa do triângulo.

    Retorna (hp, text_mask). text_mask é sempre retornada (mesmo em falha)
    para alimentar a janela de debug DEBUG_MASK_HP — é vazia 0x0 só quando
    a ROI nem chegou a ser processada.
    """
    empty = np.zeros((1, 1), dtype=np.uint8)
    if is_triangle:
        # Triângulo: pad mínimo pra não clipar o dígito, que fica deslocado
        # pro centroide do meio colorido (não pro centro do bbox).
        pad_x = pad_y = 2
    elif is_2x2:
        # 2×2 consolidado: HP de 3 dígitos (ex: "107") ocupa boa parte da
        # largura. Pad proporcional w//6 cortava os dígitos das pontas
        # (sobrava só o "1" do meio). Pad mínimo preserva o número inteiro.
        pad_x = pad_y = 2
    else:
        pad_x = max(2, w // 6)
        pad_y = max(2, h // 6)
    x1, y1 = x + pad_x, y + pad_y
    x2, y2 = x + w - pad_x, y + h - pad_y

    if x2 <= x1 or y2 <= y1:
        return 1, empty

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return 1, empty

    text_mask = _get_block_text_mask(roi, is_triangle=is_triangle)

    if np.count_nonzero(text_mask) < 5:
        return 1, text_mask

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    if is_2x2:
        # 2×2: ROI tem ~4x mais pixels que num 1×1, então o min_area
        # proporcional descartava os dígitos "0" e "7" do "107", deixando só
        # o "1". Fixo em 50 (equivalente ao 1×1 normal).
        min_area = max(3, 50)
    else:
        min_area = max(3, (x2 - x1) * (y2 - y1) // 100)

    # CORREÇÃO 1: Carrega os templates corretos do HP (Pasta: hp_digits)
    _load_hp_digit_templates()
    if not _HP_DIGIT_TEMPLATES:
        return 1, text_mask

    components = []
    for i in range(1, n_labels):
        cx, cy, cw, ch, ca = stats[i]
        if ca < min_area or ch < 4:
            continue
        crop = text_mask[cy:cy+ch, cx:cx+cw]
        components.append((cx, crop))

    if not components:
        return 1, text_mask

    components.sort(key=lambda c: c[0])
    digits_str = ""
    for _, crop in components:
        # CORREÇÃO 2: Usa a função de match correta para o tamanho do HP (20px)
        d = _match_hp_digit(crop)
        if d is not None:
            digits_str += d

    if not digits_str:
        return 1, text_mask
    return int(digits_str), text_mask

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


def _is_green_block(hsv_roi: np.ndarray) -> bool:
    """Verifica se o bloco é verde usando a faixa 'verde' de BLOCK_COLOR_RANGES."""
    if hsv_roi.size == 0:
        return False
    for name, low, high in cfg.BLOCK_COLOR_RANGES:
        if name == "verde":
            green_mask = cv2.inRange(hsv_roi, low, high)
            return np.count_nonzero(green_mask) > hsv_roi.shape[0] * hsv_roi.shape[1] * 0.30
    return False


def _qualifies_as_big_block(
    hsv: np.ndarray,                              # frame HSV inteiro; usado pra cropar a bbox e rodar as color masks
    x: int, y: int, w: int, h: int,               # bbox candidata em coords crop-relativas
    tolerance: float = 0.15,                      # margem ± em cima do tamanho ideal 2*BLOCK_SIZE
    dominance_threshold: float = 0.80,            # fracao minima da area MASCARADA que a cor top precisa cobrir
) -> bool:
    """
    Gate pra categoria "bloco 2x2 unico". As TRES checagens abaixo tem que passar:

      1. Dimensao (~ 2*BLOCK_SIZE ± 15% em width E height): descarta contornos
         que nao tem cara de 2x2 (ex: 1x2 fundido ~60x128, ou 3x3 gigante ~192x192).
      2. Cor uniforme dominante: passa TODAS as faixas de BLOCK_COLOR_RANGES no
         ROI; a cor de maior contagem tem que representar >= dominance_threshold
         (80%) do total mascarado. Se duas cores dividem o ROI significativamente
         (ex: roxo 55% + vermelho 40% da area mascarada), rejeita — sao 2+ blocos
         adjacentes de cores diferentes, nao um 2x2 unico.
      3. Contagem de numeros de HP: reusa _get_block_text_mask pra isolar o texto
         escuro dentro do bloco, agrupa os digitos em "clusters" (mesmo HP) por
         proximidade (BLOCK_SIZE/3 em X, BLOCK_SIZE/4 em Y). Um 2x2 legitimo tem
         UM cluster (o HP centralizado). Se aparecerem 2+ clusters, ha um HP por
         quadrante = 2+ blocos coloridos iguais fundidos, nao um 2x2.

    Rejeicoes sao printadas com valores medidos pra auditoria.
    """
    # --- Checagem 1: dimensao ---
    expected = 2 * cfg.BLOCK_SIZE                          # tamanho ideal de 2x2 em pixels (=128 com BLOCK_SIZE=64)
    dim_lo = int(expected * (1 - tolerance))               # limite inferior da faixa (~108 com tolerance=0.15)
    dim_hi = int(expected * (1 + tolerance))               # limite superior (~147)
    if not (dim_lo <= w <= dim_hi and dim_lo <= h <= dim_hi):  # w OU h fora da faixa -> nao eh 2x2
        if cfg.VISION_VERBOSE:
            print(
                f"[big-block reject dim] bbox=({x},{y}) {w}x{h}"
                f"  faixa=[{dim_lo}..{dim_hi}]"
            )
        return False                                       # dim errada -> ja retorna, nem chega na cor

    # --- Checagem 2: cor uniforme dominante ---
    roi = hsv[y:y + h, x:x + w]                            # recorta a bbox no frame HSV
    if roi.size == 0:                                      # bbox degenerada (fora do frame etc)
        if cfg.VISION_VERBOSE:
            print(f"[big-block reject roi-empty] bbox=({x},{y}) {w}x{h}")
        return False                                       # sem dados -> rejeita

    color_counts = []                                      # lista (nome, nnz) pra ranking depois
    for name, low, high in cfg.BLOCK_COLOR_RANGES:         # mesma lista de cores usada em detect_blocks
        mask = cv2.inRange(roi, low, high)                 # mascara binaria: pixels que caem nesta faixa HSV
        color_counts.append((name, int(np.count_nonzero(mask))))  # guarda contagem por cor

    total_masked = sum(nnz for _, nnz in color_counts)     # soma de pixels ativados em QUALQUER cor (denominador da dominancia)
    if total_masked == 0:                                  # ROI nao ativou nenhuma cor conhecida (ex: fundo preto)
        if cfg.VISION_VERBOSE:
            print(f"[big-block reject no-color] bbox=({x},{y}) {w}x{h}")
        return False                                       # sem cor -> nao eh bloco

    color_counts.sort(key=lambda t: -t[1])                 # ordena decrescente por nnz (top color primeiro)
    dominant_name, dominant_nnz = color_counts[0]          # cor com mais pixels ativados
    dominance = dominant_nnz / total_masked                # fracao do total mascarado que a cor top domina
    if dominance < dominance_threshold:                    # abaixo de 80% -> ha outra cor competindo significativamente
        if cfg.VISION_VERBOSE:
            top3 = [(n, c) for n, c in color_counts[:3] if c > 0]
            print(
                f"[big-block reject color-mix] bbox=({x},{y}) {w}x{h}"
                f"  dominante={dominant_name}={dominance:.0%}"
                f"  top3={top3}"
            )
        return False                                       # mistura de cores -> cluster de blocos, nao 2x2

    # --- Checagem 3: contagem de numeros de HP ---
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR)         # _get_block_text_mask espera BGR; ROI eh pequeno, conversao barata
    text_mask = _get_block_text_mask(roi_bgr, is_triangle=False)  # isola texto (dark pixels dentro do hull colorido)
    n_labels, _, comp_stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)  # componentes = digitos

    min_area = max(4, (w * h) // 400)                      # ruido/pontos sao <4 px; escala com o ROI
    components = []                                        # (centroid_x, centroid_y) de cada digito valido
    for i in range(1, n_labels):                           # label 0 = background, comeca em 1
        cxi, cyi, cwi, chi, cai = comp_stats[i]
        if cai < min_area or chi < 4:                      # descarta ruido pequeno e componentes finos demais
            continue
        components.append((cxi + cwi / 2.0, cyi + chi / 2.0))

    if len(components) <= 1:                               # 0 ou 1 componente -> definitivamente 1 HP so
        return True                                        # (0 acontece quando HP nao foi detectado, tolera)

    dx_max = cfg.BLOCK_SIZE / 3.0                          # ~21 px: gap max entre digitos do MESMO HP
    dy_max = cfg.BLOCK_SIZE / 4.0                          # ~16 px: linhas de digitos do MESMO HP ficam alinhadas
    clusters: List[List[Tuple[float, float]]] = []          # cada cluster representa UM numero de HP
    for cx_i, cy_i in components:                          # atribui cada componente ao cluster proximo, ou cria novo
        joined = False
        for cluster in clusters:
            if any(abs(cx_i - ccx) < dx_max and abs(cy_i - ccy) < dy_max for ccx, ccy in cluster):
                cluster.append((cx_i, cy_i))               # digito colado num membro do cluster -> mesmo HP
                joined = True
                break
        if not joined:
            clusters.append([(cx_i, cy_i)])                # muito longe -> novo HP

    if len(clusters) > 1:                                  # 2+ HPs = 2+ blocos, nao eh 2x2 unico
        cluster_centers = []                               # centroide medio de cada cluster + qtd digitos, pra debug
        for cluster in clusters:
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)
            cluster_centers.append((int(avg_x), int(avg_y), len(cluster)))
        if cfg.VISION_VERBOSE:
            print(
                f"[big-block reject hp-count] bbox=({x},{y}) {w}x{h}"
                f"  n_HPs={len(clusters)}"
                f"  centros(x,y,digitos)={cluster_centers}"
            )
        return False                                       # multiplos HPs -> multiplos blocos, rejeita

    return True                                            # passou dim + cor + hp-count -> qualifica como 2x2 unico


def detect_blocks(frame: np.ndarray) -> List[Block]:
    """Detecta todos os blocos coloridos no frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    raw_mask = cv2.inRange(hsv, cfg.BLOCK_HSV_LOW, cfg.BLOCK_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _MARGIN = 3

    # HP do bloco: leitura direta do número escrito dentro,
    # via template matching contra templates/digits/ (fundo preto, dígito branco).
    # Retorna (hp, text_mask) — a máscara é guardada em Block.hp_mask p/ debug.
    # is_tri=True ativa dilatação do hull pra triângulos (evita cortar texto na hipotenusa).
    def _hp(bx, by, bw, bh, row=-1, col=-1, is_tri=False, is_2x2=False):
        return _read_hp(frame, bx, by, bw, bh, is_triangle=is_tri, is_2x2=is_2x2)

    blocks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.MIN_BLOCK_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Power-ups são pequenos demais para ser bloco
        if w < 35 or h < 35:
            continue

        # DEBUG: rastreia contornos grandes (>= 2 blocos de área)
        _is_big = area >= cfg.BLOCK_SIZE * cfg.BLOCK_SIZE * 2
        if _is_big and cfg.VISION_VERBOSE:
            print(
                f"[contour] area={int(area)} bbox=({x},{y}) {w}x{h}"
                f"  MAX={int(cfg.MAX_BLOCK_AREA)}"
                f"  -> {'big-area' if area > cfg.MAX_BLOCK_AREA else 'normal-path'}"
            )

        # Contorno gigante (fusão morfológica de múltiplos blocos) — split por grade
        if area > cfg.MAX_BLOCK_AREA:
            # Fase 1: coleta cells sobreviventes (fill ≥ 0.40) e já classifica
            # cada uma individualmente. Nada é emitido ainda.
            survivors = []
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
                            tri_orient = _get_triangle_orientation(raw_roi)
                            if tri_orient != TriangleOrientation.UNKNOWN:
                                block_type = BlockType.TRIANGLE
                            else:
                                block_type = BlockType.SQUARE
                        else:
                            block_type = BlockType.SQUARE
                            tri_orient = TriangleOrientation.UNKNOWN
                        vertices = raw_approx.reshape(-1, 2)
                    else:
                        block_type = BlockType.SQUARE
                        tri_orient = TriangleOrientation.UNKNOWN
                        vertices = np.array([[cell_x, cell_y]])
                    survivors.append({
                        "ri": ri, "ci": ci,
                        "cell_x": cell_x, "cell_y": cell_y,
                        "cell_w": cell_w, "cell_h": cell_h,
                        "block_type": block_type,
                        "tri_orient": tri_orient,
                        "vertices": vertices,
                    })

            if cfg.VISION_VERBOSE:
                print(
                    f"[big-area phase1] area={int(area)}"
                    f"  survivors={[(7 - s['ri'], s['ci']) for s in survivors]}"
                )

            # Fase 2: tenta agrupar quadrados em blocos 2×2 da mesma cor.
            # Sem isso, um bloco grande 2×2 com HP de 3 dígitos (ex: rosa "107")
            # vira 4-5 cells separadas e cada uma le só um pedaço do número.
            survivor_set = {(s["ri"], s["ci"]) for s in survivors}
            consumed = set()
            for r_orig in range(len(cfg.LEVEL_TOPS) - 1):
                for c_orig in range(cfg.GRID_COLUMNS - 1):
                    cells_2x2 = [
                        (r_orig, c_orig), (r_orig, c_orig + 1),
                        (r_orig + 1, c_orig), (r_orig + 1, c_orig + 1),
                    ]
                    if not all(c in survivor_set for c in cells_2x2):
                        continue
                    if any(c in consumed for c in cells_2x2):
                        continue
                    bx1 = max(0, cfg.COL_LEFTS[c_orig])
                    bx2 = min(hsv.shape[1], cfg.COL_RIGHTS[c_orig + 1])
                    by1 = max(0, cfg.LEVEL_TOPS[r_orig + 1])
                    by2 = min(hsv.shape[0], cfg.LEVEL_BOTS[r_orig])
                    block_hsv = hsv[by1:by2, bx1:bx2]
                    block_area_2x2 = block_hsv.shape[0] * block_hsv.shape[1]
                    if block_area_2x2 == 0:
                        continue
                    color_ok = False
                    color_cov_2x2 = {}
                    color_nnz_2x2 = {}
                    for _name, low, high in cfg.BLOCK_COLOR_RANGES:
                        color_mask_2x2 = cv2.inRange(block_hsv, low, high)
                        nnz = int(np.count_nonzero(color_mask_2x2))
                        cov = nnz / block_area_2x2
                        color_cov_2x2[_name] = round(cov, 2)
                        color_nnz_2x2[_name] = nnz
                        if cov > 0.40:
                            color_ok = True
                    if cfg.VISION_VERBOSE:
                        print(
                            f"[big-area gate] L{7 - r_orig}-L{7 - (r_orig + 1)}"
                            f" C{c_orig}-C{c_orig + 1}"
                            f"  bbox=({bx1},{by1})-({bx2},{by2})"
                            f"  block_hsv={block_hsv.shape}"
                            f"  area={block_area_2x2}"
                            f"  nnz={color_nnz_2x2}"
                            f"  ok={color_ok}"
                        )
                    if not color_ok:
                        if cfg.VISION_VERBOSE:
                            print(
                                f"[big-area 2x2 reject] L{7 - r_orig}-L{7 - (r_orig + 1)}"
                                f" C{c_orig}-C{c_orig + 1}  color_cov={color_cov_2x2}"
                            )
                        continue
                    bw_2x2 = bx2 - bx1
                    bh_2x2 = by2 - by1
                    # Gate adicional: dim ~2x2 (±15%) + cor uniforme dominante (>=80%
                    # da area mascarada). Descarta agrupamentos onde a grade fica
                    # esticada (ex: R0 pos-ajuste) ou onde 4 cells de cores diferentes
                    # foram agrupadas por engano.
                    if not _qualifies_as_big_block(hsv, bx1, by1, bw_2x2, bh_2x2):
                        if cfg.VISION_VERBOSE:
                            print(
                                f"[big-area 2x2 reject big-block] L{7 - r_orig}-L{7 - (r_orig + 1)}"
                                f" C{c_orig}-C{c_orig + 1}  dim={bw_2x2}x{bh_2x2}"
                                f"  (motivo detalhado no [big-block reject ...] acima)"
                            )
                        continue
                    hp_2x2, hp_mask_2x2 = _hp(
                        bx1, by1, bw_2x2, bh_2x2, row=r_orig, col=c_orig,
                        is_2x2=True,
                    )
                    # Revertido do fix da ISSUE 1: 2x2 dentro de big-area usa a
                    # bbox 2x2 real (bx1..bx2, by1..by2), nao a do contorno gigante
                    # que pode cobrir 3+ celulas alem do grupo 2x2.
                    blocks.append(Block(
                        x=bx1, y=by1, w=bw_2x2, h=bh_2x2,
                        cx=bx1 + bw_2x2 / 2.0, cy=by1 + bh_2x2 / 2.0,
                        block_type=BlockType.SQUARE, hp=hp_2x2,
                        vertices=np.array(
                            [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
                        ),
                        tri_orientation=TriangleOrientation.UNKNOWN,
                        row_index=r_orig, col_index=c_orig,
                        hp_mask=hp_mask_2x2,
                    ))
                    for c in cells_2x2:
                        consumed.add(c)
                    if cfg.VISION_VERBOSE:
                        print(
                            f"[big-area 2x2] L{7 - r_orig}-L{7 - (r_orig + 1)}"
                            f" C{c_orig}-C{c_orig + 1}  hp={hp_2x2}"
                        )
                        print(
                            f"[GRID vs CONTOUR] big-area-2x2"
                            f" L{7 - r_orig}-L{7 - (r_orig + 1)} C{c_orig}-C{c_orig + 1}"
                            f" grid=({bx1},{by1},{bx2},{by2})"
                            f" contour=({x},{y},{x + w},{y + h})"
                        )

            # Fase 3: cells não consumidas viram blocos individuais.
            for s in survivors:
                if (s["ri"], s["ci"]) in consumed:
                    continue
                if cfg.VISION_VERBOSE:
                    print(
                        f"[big-area split] L{7 - s['ri']}C{s['ci']}"
                        f"  block_type={s['block_type'].value}"
                        f"  src_area={int(area)}"
                    )
                    print(
                        f"[GRID vs CONTOUR] big-area-split L{7 - s['ri']}C{s['ci']}"
                        f" grid=({s['cell_x']},{s['cell_y']},"
                        f"{s['cell_x'] + s['cell_w']},{s['cell_y'] + s['cell_h']})"
                        f" contour=({x},{y},{x + w},{y + h})"
                    )
                cell_hp, cell_hp_mask = _hp(
                    s["cell_x"], s["cell_y"], s["cell_w"], s["cell_h"],
                    row=s["ri"], col=s["ci"],
                    is_tri=(s["block_type"] == BlockType.TRIANGLE),
                )
                # Revertido do fix da ISSUE 1: cada cell usa a bbox da grade
                # dela mesma, nao a do contorno gigante que fundiu todas.
                blocks.append(Block(
                    x=s["cell_x"], y=s["cell_y"],
                    w=s["cell_w"], h=s["cell_h"],
                    cx=s["cell_x"] + s["cell_w"] / 2.0,
                    cy=s["cell_y"] + s["cell_h"] / 2.0,
                    block_type=s["block_type"], hp=cell_hp,
                    vertices=s["vertices"],
                    tri_orientation=s["tri_orient"],
                    row_index=s["ri"], col_index=s["ci"],
                    hp_mask=cell_hp_mask,
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
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx_cnt = M["m10"] / M["m00"]
                cy_cnt = M["m01"] / M["m00"]
                c_mid = _get_col_index(cx_cnt)
                r_mid = _get_row_index(cy_cnt)
                if c_left == -1:
                    c_left = c_mid
                if c_right == -1:
                    c_right = c_mid
                if r_top == -1:
                    r_top = r_mid
                if r_bot == -1:
                    r_bot = r_mid

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
                    tri_orient = _get_triangle_orientation(raw_roi)
                    if tri_orient != TriangleOrientation.UNKNOWN:
                        block_type = BlockType.TRIANGLE
                    else:
                        block_type = BlockType.SQUARE
                else:
                    block_type = BlockType.SQUARE
                    tri_orient = TriangleOrientation.UNKNOWN

                vertices = raw_approx.reshape(-1, 2)
            else:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, cfg.POLY_APPROX_FACTOR * peri, True)
                block_type, tri_orient = _classify_polygon(cnt, approx, (x, y, w, h))
                if block_type == BlockType.TRIANGLE:
                    block_type = BlockType.SQUARE
                vertices = approx.reshape(-1, 2)

            is_tri_arg = (block_type == BlockType.TRIANGLE)
            if cfg.VISION_VERBOSE:
                print(
                    f"[1x1 hp] L{7 - r_top}C{c_left}"
                    f"  block_type={block_type.value}"
                    f"  is_tri={is_tri_arg}"
                    f"  tri_orient={tri_orient.value}"
                )
            hp, hp_mask = _hp(
                x, y, w, h, row=r_top, col=c_left,
                is_tri=is_tri_arg,
            )
            blocks.append(Block(
                x=x, y=y, w=w, h=h,
                cx=x + w / 2.0, cy=y + h / 2.0,
                block_type=block_type, hp=hp,
                vertices=vertices,
                tri_orientation=tri_orient,
                row_index=r_top, col_index=c_left,
                hp_mask=hp_mask,
            ))

        elif cols_span == 2 and rows_span == 2:
            # Candidato a bloco grande 2×2 — validar que as 4 cells estão preenchidas
            # com a mesma cor (bloco real vs 3 blocos fundidos por morfologia)
            r_min_2 = min(r_top, r_bot)
            c_min_2 = min(c_left, c_right)
            # antigo: gate de cell-fill (precisa >0.50 em cada uma das 4 cells).
            # Desabilitado porque HP de 3 dígitos (ex: "107") deixa buracos
            # grandes no miolo do "0" que o MORPH_CLOSE 3×3/2it não fecha,
            # derrubando o fill abaixo do threshold e fatiando o 2×2 em 4
            # cells de HP=1. Validação fica só no gate de cor uniforme abaixo.
            is_real_2x2 = True

            # DEBUG: cell_fills só pra log (não rejeita mais)
            cell_fills_dbg = []
            for ri in range(r_min_2, r_min_2 + 2):
                for ci in range(c_min_2, c_min_2 + 2):
                    cy1 = max(0, cfg.LEVEL_TOPS[ri])
                    cy2 = min(mask.shape[0], cfg.LEVEL_BOTS[ri])
                    cx1 = max(0, cfg.COL_LEFTS[ci])
                    cx2 = min(mask.shape[1], cfg.COL_RIGHTS[ci])
                    cell_roi_dbg = mask[cy1:cy2, cx1:cx2]
                    if cell_roi_dbg.size == 0:
                        cell_fills_dbg.append(-1.0)
                    else:
                        cell_fills_dbg.append(
                            np.count_nonzero(cell_roi_dbg) / cell_roi_dbg.size
                        )

            # Verifica cor uniforme: uma única color range deve cobrir >40% de toda a área
            color_coverage_dbg = {}
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
                    coverage_ratio = (
                        np.count_nonzero(color_mask) / block_area if block_area else 0
                    )
                    color_coverage_dbg[_name] = coverage_ratio
                    if coverage_ratio > 0.40:
                        color_ok = True
                if not color_ok:
                    is_real_2x2 = False
                # Gate adicional: dimensoes precisam bater com 2x2 (com 15% de
                # tolerancia). Um 1x2 fundido (60x128) passaria no gate de cor
                # mas nao aqui.
                dim_ok = _qualifies_as_big_block(hsv, x, y, w, h)
                if not dim_ok:
                    is_real_2x2 = False
                if cfg.VISION_VERBOSE:
                    print(
                        f"[2x2 cand] L{7 - r_min_2}-L{7 - (r_min_2 + 1)} "
                        f"C{c_min_2}-C{c_min_2 + 1}"
                        f"  bbox={w}x{h}"
                        f"  cell_fills={[round(f, 2) for f in cell_fills_dbg]}"
                        f"  color_cov={ {k: round(v, 2) for k, v in color_coverage_dbg.items()} }"
                        f"  color_ok={color_ok}  dim_ok={dim_ok}"
                        f"  -> is_real_2x2={is_real_2x2}"
                    )

            if is_real_2x2:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, cfg.POLY_APPROX_FACTOR * peri, True)
                bw_2x2 = bx2 - bx1
                bh_2x2 = by2 - by1
                hp, hp_mask = _hp(bx1, by1, bw_2x2, bh_2x2, row=r_min_2, col=c_min_2, is_2x2=True)
                if cfg.VISION_VERBOSE:
                    print(
                        f"[GRID vs CONTOUR] 2x2-real"
                        f" L{7 - r_min_2}-L{7 - (r_min_2 + 1)}"
                        f" C{c_min_2}-C{c_min_2 + 1}"
                        f" grid=({bx1},{by1},{bx2},{by2})"
                        f" contour=({x},{y},{x + w},{y + h})"
                    )
                blocks.append(Block(
                    x=x, y=y, w=w, h=h,
                    cx=x + w / 2.0, cy=y + h / 2.0,
                    block_type=BlockType.SQUARE, hp=hp,
                    vertices=approx.reshape(-1, 2),
                    tri_orientation=TriangleOrientation.UNKNOWN,
                    row_index=r_min_2, col_index=c_min_2,
                    hp_mask=hp_mask,
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
                        cell_hp, cell_hp_mask = _hp(cell_x, cell_y, cell_w, cell_h, row=ri, col=ci)
                        if cfg.VISION_VERBOSE:
                            print(
                                f"[GRID vs CONTOUR] 2x2-split L{7 - ri}C{ci}"
                                f" grid=({cell_x},{cell_y},"
                                f"{cell_x + cell_w},{cell_y + cell_h})"
                                f" contour=({x},{y},{x + w},{y + h})"
                            )
                        # Revertido do fix da ISSUE 1: cada cell usa a bbox da
                        # grade dela mesma, nao a do contorno externo (que cobre
                        # varias cells fundidas e faria a fisica ver 1 bloco gigante).
                        blocks.append(Block(
                            x=cell_x, y=cell_y, w=cell_w, h=cell_h,
                            cx=cell_x + cell_w / 2.0, cy=cell_y + cell_h / 2.0,
                            block_type=BlockType.SQUARE, hp=cell_hp,
                            vertices=np.array([[cell_x, cell_y]]),
                            tri_orientation=TriangleOrientation.UNKNOWN,
                            row_index=ri, col_index=ci,
                            hp_mask=cell_hp_mask,
                        ))

        else:
            # Contornos fundidos (2×1, 1×2, etc.) — separa em blocos individuais.
            # Classifica cada cell como triângulo ou quadrado via raw_mask (mesma
            # lógica do big-area split): evita marcar como SQUARE um triângulo
            # que ficou fundido morfologicamente com um vizinho.
            r_min = min(r_top, r_bot)
            r_max = max(r_top, r_bot)
            c_min = min(c_left, c_right)
            c_max = max(c_left, c_right)
            fused_count = 0
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

                    raw_roi = raw_mask[cy1:cy2, cx1:cx2]
                    raw_cnts_cell, _ = cv2.findContours(
                        raw_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if raw_cnts_cell:
                        raw_cnt_cell = max(raw_cnts_cell, key=cv2.contourArea)
                        hull_area_cell = cv2.contourArea(cv2.convexHull(raw_cnt_cell))
                        hull_fill_cell = hull_area_cell / max(cell_w * cell_h, 1)
                        if hull_fill_cell < 0.60:
                            tri_orient_cell = _get_triangle_orientation(raw_roi)
                            if tri_orient_cell != TriangleOrientation.UNKNOWN:
                                block_type_cell = BlockType.TRIANGLE
                            else:
                                block_type_cell = BlockType.SQUARE
                        else:
                            block_type_cell = BlockType.SQUARE
                            tri_orient_cell = TriangleOrientation.UNKNOWN
                        raw_peri = cv2.arcLength(raw_cnt_cell, True)
                        raw_approx = cv2.approxPolyDP(
                            raw_cnt_cell, cfg.POLY_APPROX_FACTOR * raw_peri, True
                        )
                        raw_approx[:, :, 0] += cell_x
                        raw_approx[:, :, 1] += cell_y
                        vertices_cell = raw_approx.reshape(-1, 2)
                    else:
                        block_type_cell = BlockType.SQUARE
                        tri_orient_cell = TriangleOrientation.UNKNOWN
                        vertices_cell = np.array([[cell_x, cell_y]])

                    cell_hp, cell_hp_mask = _hp(
                        cell_x, cell_y, cell_w, cell_h, row=ri, col=ci,
                        is_tri=(block_type_cell == BlockType.TRIANGLE),
                    )
                    if cfg.VISION_VERBOSE:
                        print(
                            f"[GRID vs CONTOUR] fused L{7 - ri}C{ci}"
                            f" grid=({cell_x},{cell_y},"
                            f"{cell_x + cell_w},{cell_y + cell_h})"
                            f" contour=({x},{y},{x + w},{y + h})"
                        )
                    # Revertido do fix da ISSUE 1: cada cell usa a bbox da grade
                    # dela mesma, nao a do contorno externo fundido.
                    blocks.append(Block(
                        x=cell_x, y=cell_y, w=cell_w, h=cell_h,
                        cx=cell_x + cell_w / 2.0, cy=cell_y + cell_h / 2.0,
                        block_type=block_type_cell, hp=cell_hp,
                        vertices=vertices_cell,
                        tri_orientation=tri_orient_cell,
                        row_index=ri, col_index=ci,
                        hp_mask=cell_hp_mask,
                    ))
                    fused_count += 1
            if cfg.VISION_VERBOSE:
                print(
                    f"[fused] cols={cols_span} rows={rows_span}"
                    f"  bbox={w}x{h} src_area={int(area)}"
                    f"  L{7 - r_min}-L{7 - r_max}C{c_min}-C{c_max}"
                    f"  -> {fused_count} blocos"
                )

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

        if b.block_type == BlockType.TRIANGLE and len(b.vertices) >= 3:
            pts = b.vertices.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(debug, [pts], True, color, 2)
        else:
            cv2.rectangle(debug, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        # label = f"{tipo} hp={b.hp} L{7 - b.row_index}C{b.col_index}"
        label = f"{b.hp} L{7 - b.row_index}C{b.col_index}"
        cx_label = b.x + b.w // 2
        cy_label = b.y + b.h // 2

        tamanho_texto = 0.45

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, 2)
        
        # Escreve valor dentro dos objetos
        eixoX = (cx_label - tw // 2)
        eixoY = (cy_label + th // 2) + 20
        cv2.putText(debug, label, (eixoX, eixoY), # Preto (contorno)
                    cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (0, 0, 0), 2)
        cv2.putText(debug, label, (eixoX, eixoY), # cor especifica
                    cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (255, 255, 255), 1)

    # _POWERUP_DEBUG_LABEL = {
    #     PowerUpType.EXTRA_BALL: "+",
    #     PowerUpType.LASER_H:    "<>",
    #     PowerUpType.LASER_V:    "^v",
    #     PowerUpType.BONUS_B:    "B",
    #     PowerUpType.UNKNOWN:    "?",
    # }
    for p in powerups:
        cv2.circle(debug, (int(p.cx), int(p.cy)), int(p.radius), (255, 255, 0), 2)
        # cv2.putText(
        #     debug,
        #     _POWERUP_DEBUG_LABEL[p.power_up_type],
        #     (int(p.cx) - 5, int(p.cy) + 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 255, 0),
        #     2,
        # )

    if ball_pos:
        cv2.circle(debug, (int(ball_x), int(ball_y)), cfg.BALL_RADIUS, (0, 0, 255), 2)

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
