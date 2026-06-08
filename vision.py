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

import config as cfg

# ─────────────────────────────────────────────
# TIPOS DE DADOS
# ─────────────────────────────────────────────


class BlockType(Enum):
    SQUARE = "square"
    TRIANGLE = "triangle"
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

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Retorna (x, y, w, h) para AABB collision."""
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)


@dataclass
class PowerUp:
    """Representa um power-up circular (+)."""

    cx: float
    cy: float
    radius: float


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
    approx: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Tuple[BlockType, TriangleOrientation]:
    """Classifica o polígono como quadrado ou triângulo e determina orientação."""
    n_vertices = len(approx)

    if n_vertices == cfg.TRIANGLE_VERTICES:
        orientation = _get_triangle_orientation(approx, bbox)
        return BlockType.TRIANGLE, orientation

    elif n_vertices == cfg.SQUARE_VERTICES:
        return BlockType.SQUARE, TriangleOrientation.UNKNOWN

    else:
        # Se tem muitos vértices mas é retangular, ainda é quadrado
        x, y, w, h = bbox
        aspect = w / max(h, 1)
        if 0.7 < aspect < 1.4:
            return BlockType.SQUARE, TriangleOrientation.UNKNOWN
        return BlockType.UNKNOWN, TriangleOrientation.UNKNOWN


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


def _read_hp_ocr(frame_gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """
    Lê o número de HP do bloco via OCR (pytesseract).
    O número é escuro sobre fundo colorido.
    Fallback para estimativa se OCR falhar.
    """
    try:
        import pytesseract
    except ImportError:
        return _read_hp_fallback(frame_gray, x, y, w, h)

    # Recorta a região central do bloco (onde fica o número)
    pad_x, pad_y = w // 5, h // 5
    roi = frame_gray[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return 1

    # Pré-processamento para OCR
    # 1. Redimensiona para altura fixa (melhora precisão do tesseract)
    target_h = 40
    scale = target_h / roi.shape[0]
    roi_big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Binariza — número escuro sobre fundo claro
    _, binary = cv2.threshold(roi_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Limpeza
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. Adiciona borda branca (tesseract precisa de margem)
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # OCR com config para dígitos apenas
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        text = pytesseract.image_to_string(binary, config=config).strip()
        if text and text.isdigit():
            hp = int(text)
            if 1 <= hp <= 999:
                return hp
    except Exception:
        pass

    # Fallback se OCR falhou
    return _read_hp_fallback(frame_gray, x, y, w, h)


def _read_hp_fallback(frame_gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """
    Fallback: estimativa de HP por contagem de componentes conexos.
    Usado se pytesseract não está instalado ou falhou.
    """
    pad_x, pad_y = w // 4, h // 4
    roi = frame_gray[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

    if roi.size == 0:
        return 1

    _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
    n_labels, _ = cv2.connectedComponents(binary)
    hp = max(1, n_labels - 1)
    return min(hp, 99)


def detect_blocks(frame: np.ndarray) -> List[Block]:
    """Detecta todos os blocos coloridos no frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Máscara para blocos laranja
    mask = cv2.inRange(hsv, cfg.BLOCK_HSV_LOW, cfg.BLOCK_HSV_HIGH)

    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.MIN_BLOCK_AREA or area > cfg.MAX_BLOCK_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Aproximação poligonal para classificar forma
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, cfg.POLY_APPROX_FACTOR * peri, True)

        block_type, tri_orient = _classify_polygon(approx, (x, y, w, h))
        hp = _read_hp_ocr(gray, x, y, w, h)

        block = Block(
            x=x,
            y=y,
            w=w,
            h=h,
            cx=x + w / 2.0,
            cy=y + h / 2.0,
            block_type=block_type,
            hp=hp,
            vertices=approx.reshape(-1, 2),
            tri_orientation=tri_orient,
        )
        blocks.append(block)

    return blocks


# ─────────────────────────────────────────────
# DETECÇÃO DE POWER-UPS
# ─────────────────────────────────────────────


def detect_powerups(frame: np.ndarray) -> List[PowerUp]:
    """Detecta power-ups circulares (+ ciano)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.POWERUP_HSV_LOW, cfg.POWERUP_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    powerups = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        # Verificar circularidade
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)

        if circularity > 0.6:  # é circular o suficiente
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            powerups.append(PowerUp(cx=cx, cy=cy, radius=radius))

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
            game_over=True,
            white_ratio=white_ratio,
            raw_frame=frame,
            debug_frame=frame.copy(),
        )

    blocks = detect_blocks(frame)
    powerups = detect_powerups(frame)
    ball_pos = detect_ball(frame)
    n_balls = count_balls(frame)

    ball_x = ball_pos[0] if ball_pos else None
    ball_y = ball_pos[1] if ball_pos else None

    # Frame de debug com anotações
    debug = frame.copy()
    for b in blocks:
        color = (0, 255, 0) if b.block_type == BlockType.SQUARE else (0, 255, 255)
        cv2.rectangle(debug, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        label = f"{b.block_type.value} hp={b.hp}"
        cv2.putText(
            debug, label, (b.x, b.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    for p in powerups:
        cv2.circle(debug, (int(p.cx), int(p.cy)), int(p.radius), (255, 255, 0), 2)
        cv2.putText(
            debug,
            "+",
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
