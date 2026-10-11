"""
visualizar_masks.py — Janelas de debug ao vivo, em thread daemon.

╔══════════════════════════════════════════════════════════════════════════╗
║  PARA LIGAR/DESLIGAR UMA JANELA: mexa só no dicionário JANELAS abaixo.   ║
║  Trocar True por False já pula o desenho E o cálculo daquele painel.     ║
╚══════════════════════════════════════════════════════════════════════════╝

Cada janela é um "painel": um nome, uma chave no dicionário JANELAS e uma
função que devolve a imagem (ou None quando não há o que mostrar). O loop
percorre a lista _PAINEIS e não sabe nada sobre o conteúdo de cada um — então
adicionar ou remover uma janela é mexer num lugar só.

Painéis disponíveis:

── Visão: o que o bot enxerga ──

pipeline     "Mask: Pipeline (raw vs morfologia)"
             A máscara que a DETECÇÃO realmente usa, nas duas metades: o
             inRange cru e o resultado do CLOSE x2 + OPEN x1, com os pixels
             que a morfologia inventou em magenta. É onde se vê por que dois
             blocos viraram um contorno só.

blocos       "Mask: Blocos (todas as cores)"
             OR das 11 faixas de BLOCK_COLOR_RANGES. CUIDADO: não é a máscara
             da detecção — é um subconjunto dela (o pipeline aceita ~46% mais
             do espaço HSV). Serve pra ver quais cores estão nomeadas, que é o
             que alimenta a leitura de HP e o gate 2x2. Para depurar detecção,
             use o painel "pipeline".

powerups     "Mask: Power-ups"
             As 3 faixas (laser azul, bola extra ciano, moeda B ouro) mais o
             que o detect_powerups aceitou, com raio e tipo. Power-ups não
             aparecem em nenhum dos painéis de bloco.

bolas        "Mask: Contagem de bolas"
             Strip da BALL_COUNT_REGION + o número lido pelo OCR.
             É o painel MAIS CARO: roda template matching a cada frame.
             (antigo: era também o único que disputava o mss global de
             vision.py com a thread principal; agora toda captura passa pelo
             tela.py, que usa uma instância por thread.)

fim_rodada   "Mask: Fim de rodada"
             As 4 regiões rosa + status (n/4).

── Grade e leitura ──

grid         "Grid: Blocos por fileira"
             debug_frame de scan_game com as linhas de separação e a
             contagem por fileira. Alimentado pelo loop principal via
             update_level_display() — fica vazio até o primeiro scan.

hp           "OCR: HP lido por bloco"
             Mosaico 7x7 das text_mask, cada tile com o HP que o OCR
             extraiu daquele bloco. Também depende de update_level_display().

gate2x2      "Gate: Checagens do bloco 2x2"
             Cada candidato a bloco grande com os 4 valores medidos
             (dimensão, preenchimento, dominância de cor, nº de HPs) e o
             veredito. Os números vêm de vision.GATE_2X2_DIAG, preenchido
             pelo próprio gate — não há lógica duplicada aqui.

── Decisão ──

trajetoria   "Bot: Trajetoria escolhida"
             O caminho do ângulo escolhido sobre o frame do scan, com os
             pontos de colisão. Alimentado por update_shot_display().

angulos      "Bot: Leque de angulos por score"
             Um raio por ângulo simulado, colorido pelo score (azul = ruim,
             vermelho = bom); o escolhido em branco. Mostra se a decisão foi
             folgada ou empate apertado — é o instrumento pra calibrar
             W_ROW0_SURVIVE.

Pressione 'q' na janela para sair.

NOTA: cv2.putText só renderiza ASCII. Texto desenhado nas janelas não pode
levar acento nem travessão (viram '???').
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg
import tela  # antigo: import mss (captura direta)

# ═════════════════════════════════════════════════════════════════════════
# LIGA / DESLIGA DAS JANELAS  ←── é aqui que você mexe
# ═════════════════════════════════════════════════════════════════════════

JANELAS = {
    # ── Grade e leitura ──
    "grid": True,###  # blocos detectados sobre a grade
    "hp": True,  # text_mask + HP lido por bloco

    # ── Decisão ──
    "trajetoria": True,  # caminho do angulo escolhido
    "angulos": True,  # leque de todos os angulos, colorido por score

    # ── Visão: o que o bot enxerga ──
    "blocos": False, #### # OR das 11 faixas nomeadas (nao e a mask da deteccao!)
    "powerups": False, #### # as 3 faixas de power-up + o que foi detectado
    "bolas": False, #### # contagem de bolas (OCR)
    "pipeline": False,  # mask REAL da deteccao, antes/depois da morfologia
    "fim_rodada": False,  # as 4 regioes rosa
    "gate2x2": False,  # checagens do gate de bloco grande
    
}


# ═════════════════════════════════════════════════════════════════════════
# CONSTANTES DE DESENHO
# ═════════════════════════════════════════════════════════════════════════

SCALE = 1.2  # escala do thumbnail das masks de tela cheia
BLOCK_SCALE = 0.80  # escala do overlay de grid
_HP_TILE = 70  # px por tile no mosaico do painel "hp"
_REFRESH_MS = 30  # intervalo do waitKey (~33 fps)

_GRAY = (128, 128, 128)
_WHITE = (220, 220, 220)
_VERDE = (0, 220, 0)
_VERMELHO = (0, 0, 220)


# ═════════════════════════════════════════════════════════════════════════
# ESTADO COMPARTILHADO — o loop principal escreve, a thread de masks lê
# ═════════════════════════════════════════════════════════════════════════

_shared_debug_frame = None
_shared_blocks = []
_shared_ball_count = 0
_shared_gate2x2 = []  # snapshot de vision.GATE_2X2_DIAG do último scan

_shared_best = None  # SimResult do ângulo escolhido
_shared_all_results = []  # SimResult de todos os ângulos
_shared_ball_xy = None  # (x, y) da bola no frame
# Frame em que _shared_best foi simulado. Não dá pra usar _shared_debug_frame:
# o loop atualiza ele com o scan do turno NOVO antes de terminar a simulação, e
# por ~2s a trajetória do turno anterior aparecia atravessando os blocos que
# desceram uma fileira.
_shared_shot_frame = None


def update_level_display(debug_frame, blocks, ball_count):
    """Chamado pelo loop principal a cada scan. Alimenta os painéis grid e hp."""
    global _shared_debug_frame, _shared_blocks, _shared_ball_count, _shared_gate2x2
    from vision import GATE_2X2_DIAG

    _shared_debug_frame = debug_frame
    _shared_blocks = blocks
    _shared_ball_count = ball_count
    # Cópia: a lista original é limpa pelo detect_blocks do próximo scan, na
    # thread principal, enquanto esta thread pode estar lendo.
    _shared_gate2x2 = list(GATE_2X2_DIAG)


def update_shot_display(best, all_results, ball_xy, frame):
    """
    Chamado pelo loop principal após find_best_angle. Alimenta os painéis
    trajetoria e angulos. Só guarda referências — é barato o bastante pra
    rodar todo turno, mesmo com os painéis desligados.

    `frame` é o debug_frame do scan que foi simulado: os dois painéis desenham
    sobre ele, então trajetória e tabuleiro são sempre do mesmo turno.
    """
    global _shared_best, _shared_all_results, _shared_ball_xy, _shared_shot_frame
    _shared_best = best
    _shared_all_results = all_results
    _shared_ball_xy = ball_xy
    _shared_shot_frame = frame


# ═════════════════════════════════════════════════════════════════════════
# HELPERS DE DESENHO
# ═════════════════════════════════════════════════════════════════════════


def _texto(img, texto, org, cor, escala=0.5):
    """Texto com contorno preto, pra ficar legível sobre qualquer fundo."""
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1)


@dataclass
class _Ctx:
    """O que todo painel recebe pra se desenhar."""

    # antigo: sct (instância mss própria desta thread) — agora tela.capturar,
    # que no backend real já usa uma instância por thread
    thumb_w: int
    thumb_h: int

    def grab(self, regiao) -> np.ndarray:
        """Captura uma região (x, y, w, h) da tela e devolve BGR."""
        return tela.capturar_regiao(regiao)

    def grab_hsv(self, regiao):
        """Igual ao grab, mas já devolve (bgr, hsv)."""
        bgr = self.grab(regiao)
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


# ═════════════════════════════════════════════════════════════════════════
# PAINÉIS — um por janela. Devolvem a imagem, ou None se não há o que mostrar.
# ═════════════════════════════════════════════════════════════════════════


def _painel_blocos(ctx: _Ctx) -> np.ndarray:
    """OR de todas as faixas de cor de bloco, sobre a GAME_REGION."""
    _, hsv = ctx.grab_hsv(cfg.GAME_REGION)

    combinada = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for _nome, low, high in cfg.BLOCK_COLOR_RANGES:
        combinada = cv2.bitwise_or(combinada, cv2.inRange(hsv, low, high))

    thumb = cv2.resize(
        cv2.cvtColor(combinada, cv2.COLOR_GRAY2BGR),
        (ctx.thumb_w, ctx.thumb_h),
        interpolation=cv2.INTER_AREA,
    )
    _texto(thumb, "COMBINADA", (4, 14), _VERDE, escala=0.4)
    return thumb


def _painel_bolas(ctx: _Ctx) -> np.ndarray:
    """Strip da região de contagem + o número que o OCR leu."""
    # antigo: read_ball_count_template() usava o mss GLOBAL de vision.py,
    # disputado com a thread principal; agora captura pelo tela.py (uma
    # instância de mss por thread).
    from vision import read_ball_count_template

    _, hsv = ctx.grab_hsv(cfg.BALL_COUNT_REGION)
    mask = cv2.inRange(hsv, cfg.BALL_COUNT_HSV_LOW, cfg.BALL_COUNT_HSV_HIGH)

    _, _, brw, brh = cfg.BALL_COUNT_REGION
    strip_h = max(50, int(brh * ctx.thumb_w / brw))
    strip = cv2.cvtColor(
        cv2.resize(mask, (ctx.thumb_w, strip_h), interpolation=cv2.INTER_AREA),
        cv2.COLOR_GRAY2BGR,
    )

    lido = read_ball_count_template()
    label = f"Bolas: {lido}" if lido > 0 else f"Bolas: {_shared_ball_count} (visual)"
    _texto(strip, label, (6, strip_h - 6), _VERDE, escala=0.6)
    return strip


def _painel_fim_rodada(ctx: _Ctx) -> Optional[np.ndarray]:
    """As 4 regiões indicadoras rosa, lado a lado, + status."""
    if not cfg.ROUND_END_REGIONS:
        return None

    pedacos = []
    acertos = 0
    altura = 40
    for regiao in cfg.ROUND_END_REGIONS:
        _, hsv = ctx.grab_hsv(regiao)
        mask = cv2.inRange(hsv, cfg.ROUND_END_HSV_LOW, cfg.ROUND_END_HSV_HIGH)
        if np.count_nonzero(mask) > mask.size * 0.1:
            acertos += 1
        _, _, rew, reh = regiao
        largura = int(rew * altura / max(reh, 1))
        pedacos.append(
            cv2.cvtColor(
                cv2.resize(mask, (largura, altura), interpolation=cv2.INTER_AREA),
                cv2.COLOR_GRAY2BGR,
            )
        )

    gap = np.zeros((altura, 4, 3), dtype=np.uint8)
    linha = pedacos[0]
    for p in pedacos[1:]:
        linha = np.hstack([linha, gap, p])
    strip = cv2.resize(linha, (ctx.thumb_w, altura), interpolation=cv2.INTER_AREA)

    # Mesmo critério de vision.detect_round_end(): >=2 das 4 regiões acesas
    acabou = acertos >= 2
    status = "RODADA ACABOU" if acabou else f"Aguardando... ({acertos}/4)"
    _texto(strip, status, (6, 14), _VERDE if acabou else _VERMELHO, escala=0.45)
    return strip


def _painel_pipeline(ctx: _Ctx) -> np.ndarray:
    """
    A máscara que a DETECÇÃO realmente usa, antes e depois da morfologia.

    Reproduz exatamente as 3 primeiras linhas de vision.detect_blocks. Duas
    diferenças em relação ao painel "blocos", e as duas importam:

      1. Usa BLOCK_HSV_LOW/HIGH (faixa larga), não o OR de BLOCK_COLOR_RANGES.
         O pipeline aceita ~46% mais do espaço HSV que as 11 faixas nomeadas —
         bloco de cor fora delas aparece aqui e não lá.
      2. Mostra o efeito do CLOSE x2 + OPEN x1. É a morfologia que funde blocos
         vizinhos num contorno gigante e dispara o caminho de big-area split,
         então "por que dois blocos viraram um" se responde comparando as
         metades.
    """
    _, hsv = ctx.grab_hsv(cfg.GAME_REGION)

    raw = cv2.inRange(hsv, cfg.BLOCK_HSV_LOW, cfg.BLOCK_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pos = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    pos = cv2.morphologyEx(pos, cv2.MORPH_OPEN, kernel, iterations=1)

    # Ganhou pixel na morfologia = magenta; o resto fica branco
    ganho = cv2.subtract(pos, raw)
    pos_bgr = cv2.cvtColor(pos, cv2.COLOR_GRAY2BGR)
    pos_bgr[ganho > 0] = (200, 0, 200)

    metade = ctx.thumb_w // 2
    alt = int(ctx.thumb_h * 0.8)
    esq = cv2.resize(
        cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), (metade, alt), cv2.INTER_AREA
    )
    dir_ = cv2.resize(pos_bgr, (metade, alt), cv2.INTER_AREA)
    _texto(esq, "RAW (inRange)", (4, 14), _VERDE, escala=0.4)
    _texto(dir_, "POS-MORFOLOGIA", (4, 14), _VERDE, escala=0.4)
    _texto(dir_, "magenta = fundido", (4, 30), (200, 0, 200), escala=0.35)

    canvas = np.hstack([esq, dir_])
    cv2.line(canvas, (metade, 0), (metade, alt), _WHITE, 1)
    return canvas


def _painel_powerups(ctx: _Ctx) -> np.ndarray:
    """
    As 3 máscaras de power-up, cada uma na sua cor, mais o que o
    detect_powerups aceitou (círculo branco + tipo).

    O painel "blocos"/"pipeline" NÃO mostra power-ups: eles têm faixas HSV
    próprias. Como W_EXTRA_BALL=30 e W_BONUS_B=5 pesam na decisão, vale ver se
    o power-up foi sequer detectado antes de culpar o scoring.
    """
    from vision import detect_powerups

    bgr, hsv = ctx.grab_hsv(cfg.GAME_REGION)

    faixas = [
        ("laser", cfg.POWERUP_LASER_HSV_LOW, cfg.POWERUP_LASER_HSV_HIGH, (235, 139, 0)),
        (
            "bola extra",
            cfg.POWERUP_EXTRA_BALL_HSV_LOW,
            cfg.POWERUP_EXTRA_BALL_HSV_HIGH,
            (211, 237, 0),
        ),
        ("moeda B", cfg.POWERUP_B_HSV_LOW, cfg.POWERUP_B_HSV_HIGH, (57, 213, 255)),
        (
            "direcao aleatoria",
            cfg.POWERUP_RANDOM_HSV_LOW,
            cfg.POWERUP_RANDOM_HSV_HIGH,
            (240, 48, 155),
        ),
    ]

    canvas = np.zeros_like(bgr)
    for _nome, low, high, cor in faixas:
        canvas[cv2.inRange(hsv, low, high) > 0] = cor

    achados = detect_powerups(bgr)
    for pu in achados:
        cv2.circle(canvas, (int(pu.cx), int(pu.cy)), int(pu.radius) + 3, _WHITE, 1)
        _texto(
            canvas,
            pu.power_up_type.value,
            (int(pu.cx) - 18, int(pu.cy) - int(pu.radius) - 6),
            _WHITE,
            escala=0.35,
        )

    thumb = cv2.resize(canvas, (ctx.thumb_w, ctx.thumb_h), interpolation=cv2.INTER_AREA)
    _texto(thumb, f"detectados: {len(achados)}", (4, 14), _VERDE, escala=0.45)
    for i, (nome, _l, _h, cor) in enumerate(faixas):
        _texto(thumb, nome, (4, 32 + i * 15), cor, escala=0.38)
    return thumb


def _painel_trajetoria(ctx: _Ctx) -> Optional[np.ndarray]:
    """
    A trajetória do ângulo escolhido, sobre o frame em que ela foi simulada.

    Antes isto era uma janela criada pela thread PRINCIPAL sob --debug, o que
    punha duas threads chamando imshow/waitKey — o HighGUI do OpenCV não é
    thread-safe. Como painel, o desenho tem um dono só.
    """
    # antigo: desenhava sobre _shared_debug_frame (scan mais recente)
    if _shared_best is None or _shared_shot_frame is None:
        return None
    if not _shared_best.path:
        return None

    vis = _shared_shot_frame.copy()
    pts = np.array(_shared_best.path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)

    for cx, cy in _shared_best.collisions:
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    if _shared_ball_xy:
        cv2.circle(
            vis, (int(_shared_ball_xy[0]), int(_shared_ball_xy[1])), 5, _WHITE, 1
        )

    b = _shared_best
    _texto(vis, f"{b.angle:.0f}deg  score {b.score:.0f}", (4, 16), _VERDE, escala=0.5)
    _texto(
        vis,
        f"hits {b.total_hits}  destroi {b.blocks_destroyed}",
        (4, 34),
        _WHITE,
        escala=0.42,
    )
    return vis


def _painel_angulos(ctx: _Ctx) -> Optional[np.ndarray]:
    """
    Leque de todos os ângulos simulados, colorido pelo score (azul = ruim,
    vermelho = bom). O escolhido sai em branco e grosso.

    Serve pra ver se a decisão foi folgada ou um empate apertado, e se a faixa
    ANGLE_MIN..ANGLE_MAX está deixando algo bom de fora. É o instrumento pra
    calibrar W_ROW0_SURVIVE: com o peso desligado a curva tem um formato, com
    ele ligado outro.
    """
    if not _shared_all_results or _shared_ball_xy is None:
        return None

    # antigo: _shared_debug_frame (scan mais recente, podia ser do turno seguinte)
    fundo = (
        _shared_shot_frame
        if _shared_shot_frame is not None
        else np.zeros((cfg.GAME_REGION[3], cfg.GAME_REGION[2], 3), np.uint8)
    )
    vis = (fundo * 0.35).astype(np.uint8)

    scores = [r.score for r in _shared_all_results]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    bx, by = _shared_ball_xy
    comprimento = cfg.GAME_REGION[3] * 0.42

    melhor = max(_shared_all_results, key=lambda r: r.score)
    for r in sorted(_shared_all_results, key=lambda r: r.score):
        norm = (r.score - lo) / span
        cor = cv2.applyColorMap(np.uint8([[int(norm * 255)]]), cv2.COLORMAP_JET).ravel()
        rad = np.radians(r.angle)
        fim = (int(bx + np.cos(rad) * comprimento), int(by - np.sin(rad) * comprimento))
        cv2.line(vis, (int(bx), int(by)), fim, cor.tolist(), 1, cv2.LINE_AA)

    rad = np.radians(melhor.angle)
    fim = (
        int(bx + np.cos(rad) * comprimento * 1.15),
        int(by - np.sin(rad) * comprimento * 1.15),
    )
    cv2.line(vis, (int(bx), int(by)), fim, _WHITE, 2, cv2.LINE_AA)

    _texto(vis, f"melhor {melhor.angle:.0f}deg", (4, 16), _WHITE, escala=0.5)
    _texto(vis, f"score {lo:.0f} .. {hi:.0f}", (4, 34), _VERDE, escala=0.42)
    return vis


def _painel_gate2x2(ctx: _Ctx) -> Optional[np.ndarray]:
    """
    As checagens do gate de bloco 2x2, com os valores MEDIDOS pelo próprio
    gate (vision.GATE_2X2_DIAG) — sem lógica duplicada aqui.

    É o painel que teria matado em segundos o falso positivo de "2 blocos 1x1
    roxos + 1 triângulo rosa": a dominância deu 0.8045 contra um limiar de
    0.80, passando por 0.5%.
    """
    if not _shared_gate2x2:
        return None

    linha_h = 18
    canvas = np.zeros((30 + linha_h * (len(_shared_gate2x2) + 1), 460, 3), np.uint8)
    # Só ASCII no putText: o OpenCV não renderiza acentos nem travessão (vira '???')
    _texto(canvas, "GATE 2x2 - candidatos do ultimo scan", (6, 16), _VERDE, escala=0.45)
    _texto(
        canvas,
        "bbox            dim   fill    dom  HPs   veredito",
        (6, 16 + linha_h),
        _GRAY,
        escala=0.4,
    )

    for i, d in enumerate(_shared_gate2x2):
        x, y, w, h = d["bbox"]
        fill = f"{d['fill']:.2f}" if d["fill"] is not None else "  - "
        dom = f"{d['dominancia']:.2f}" if d["dominancia"] is not None else "  - "
        hps = str(d["n_hps"]) if d["n_hps"] is not None else "-"
        dim = "ok" if d["dim_ok"] else "no"
        ok = d["veredito"]
        txt = (
            f"{x:3d},{y:3d} {w:3d}x{h:3d}  {dim:3} {fill:>5} {dom:>6} {hps:>3}   "
            f"{'ACEITO' if ok else 'rejeita: ' + d['motivo']}"
        )
        _texto(
            canvas,
            txt,
            (6, 16 + linha_h * (i + 2)),
            _VERDE if ok else _VERMELHO,
            escala=0.38,
        )
    return canvas


def _painel_grid(ctx: _Ctx) -> Optional[np.ndarray]:
    """debug_frame do último scan, com a grade e a contagem por fileira."""
    if _shared_debug_frame is None:
        return None  # ainda não houve scan
    return _draw_levels_overlay(_shared_debug_frame, _shared_blocks)


def _painel_hp(ctx: _Ctx) -> Optional[np.ndarray]:
    """Mosaico com a text_mask e o HP lido de cada bloco."""
    if not _shared_blocks:
        return None  # ainda não houve scan
    return _build_hp_debug_mosaic(_shared_blocks)


# ═════════════════════════════════════════════════════════════════════════
# REGISTRO — ligar a chave de JANELAS ao título e ao construtor
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _Painel:
    chave: str  # chave em JANELAS
    titulo: str  # nome da janela do OpenCV
    construir: Callable[[_Ctx], Optional[np.ndarray]]


_PAINEIS = (
    _Painel("pipeline", "Mask: Pipeline (raw vs morfologia)", _painel_pipeline),
    _Painel("blocos", "Mask: Blocos (todas as cores)", _painel_blocos),
    _Painel("powerups", "Mask: Power-ups", _painel_powerups),
    _Painel("bolas", "Mask: Contagem de bolas", _painel_bolas),
    _Painel("fim_rodada", "Mask: Fim de rodada", _painel_fim_rodada),
    _Painel("grid", "Grid: Blocos por fileira", _painel_grid),
    _Painel("hp", "OCR: HP lido por bloco", _painel_hp),
    _Painel("gate2x2", "Gate: Checagens do bloco 2x2", _painel_gate2x2),
    _Painel("trajetoria", "Bot: Trajetoria escolhida", _painel_trajetoria),
    _Painel("angulos", "Bot: Leque de angulos por score", _painel_angulos),
)


# ═════════════════════════════════════════════════════════════════════════
# CONSTRUTORES DE IMAGEM (usados pelos painéis grid e hp)
# ═════════════════════════════════════════════════════════════════════════


def _build_hp_debug_mosaic(blocks) -> np.ndarray:
    """
    Tila a text_mask de cada bloco (Block.hp_mask) numa grade row x col que
    espelha a disposicao do jogo: row_index=6 (topo do jogo) em cima,
    row_index=0 (perigosa) embaixo. Cada tile mostra o HP que o OCR leu.
    """
    rows = len(cfg.LEVEL_TOPS)  # 7
    cols = cfg.GRID_COLUMNS  # 7
    tile = _HP_TILE
    canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)

    for b in blocks:
        if b.row_index < 0 or b.col_index < 0:
            continue
        canvas_row = (rows - 1) - b.row_index  # inverte: topo do jogo no topo da janela
        y0 = canvas_row * tile
        x0 = b.col_index * tile

        mask = b.hp_mask
        if mask.size == 0 or mask.shape[0] < 2 or mask.shape[1] < 2:
            cell = np.zeros((tile, tile), dtype=np.uint8)
        else:
            cell = cv2.resize(mask, (tile, tile), interpolation=cv2.INTER_NEAREST)
        canvas[y0 : y0 + tile, x0 : x0 + tile] = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)

        _texto(canvas, str(b.hp), (x0 + 4, y0 + 16), _VERDE)

    # Grade separando tiles
    for i in range(1, rows):
        cv2.line(canvas, (0, i * tile), (cols * tile, i * tile), (50, 50, 50), 1)
    for j in range(1, cols):
        cv2.line(canvas, (j * tile, 0), (j * tile, rows * tile), (50, 50, 50), 1)

    return canvas


def _draw_levels_overlay(debug_frame: np.ndarray, blocks) -> np.ndarray:
    """
    Sobrepõe ao debug_frame:
      - grade cinza separando linhas e colunas
      - label de coluna (C0–C6) e timestamp do scan
    """
    s = BLOCK_SCALE
    h, w = debug_frame.shape[:2]
    vis = cv2.resize(debug_frame, (int(w * s), int(h * s)))
    W, H = vis.shape[1], vis.shape[0]

    row_counts: dict = {}
    for b in blocks:
        row_counts[b.row_index] = row_counts.get(b.row_index, 0) + 1

    # ── Linhas horizontais (separação de rows) ──
    for row_idx in range(7):
        top_y = int(cfg.LEVEL_TOPS[row_idx] * s)
        bot_y = int(cfg.LEVEL_BOTS[row_idx] * s)
        cv2.line(vis, (0, top_y), (W, top_y), _GRAY, 1, cv2.LINE_AA)

        # Label da fileira desligado — poluía o frame. Pra reativar, descomente:
        # display_level = 7 - row_idx
        # count = row_counts.get(row_idx, 0)
        # suffix = " CRIT" if row_idx == 0 else ""
        # mid_y = (top_y + bot_y) // 2 + 5
        # _texto(vis, f"L{display_level}{suffix}({count})", (3, mid_y), _WHITE, 0.32)

    # Linha inferior do último row
    bot0 = int(cfg.LEVEL_BOTS[0] * s)
    cv2.line(vis, (0, bot0), (W, bot0), _GRAY, 1, cv2.LINE_AA)

    # ── Linhas verticais (separação de colunas) ──
    grid_top_y = int(cfg.LEVEL_TOPS[6] * s)
    for col_idx in range(cfg.GRID_COLUMNS):
        lx = int(cfg.COL_LEFTS[col_idx] * s)
        rx = int(cfg.COL_RIGHTS[col_idx] * s)
        cv2.line(vis, (lx, grid_top_y), (lx, bot0), _GRAY, 1, cv2.LINE_AA)
        cv2.line(vis, (rx, grid_top_y), (rx, bot0), _GRAY, 1, cv2.LINE_AA)
        _texto(vis, f"C{col_idx}", ((lx + rx) // 2 - 8, grid_top_y - 3), _WHITE, 0.3)

    cv2.putText(
        vis,
        f"scan {time.strftime('%H:%M:%S')}",
        (3, H - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (100, 100, 100),
        1,
    )
    return vis


# ═════════════════════════════════════════════════════════════════════════
# LOOP
# ═════════════════════════════════════════════════════════════════════════


def painels_ativos():
    """Os painéis ligados em JANELAS, na ordem do registro."""
    return [p for p in _PAINEIS if JANELAS.get(p.chave, False)]


def run_masks_window(stop_event=None):
    """
    Loop das janelas de debug. Roda em thread daemon, iniciada por main.py.
    Tecla 'q' (com foco numa das janelas) encerra.

    Não sabe nada sobre o conteúdo dos painéis: percorre os que estão ligados,
    pede a imagem de cada um e mostra. Painel que devolve None é pulado — é
    assim que grid e hp ficam quietos até o primeiro scan chegar.
    """
    ativos = painels_ativos()
    if not ativos:
        return  # tudo desligado em JANELAS — nem abre a thread de desenho

    rw, rh = cfg.GAME_REGION[2], cfg.GAME_REGION[3]
    ctx = _Ctx(
        thumb_w=int(rw * SCALE),
        thumb_h=int(rh * SCALE),
    )

    try:
        while not (stop_event and stop_event.is_set()):
            for painel in ativos:
                img = painel.construir(ctx)
                if img is None:
                    continue  # painel sem nada a mostrar neste frame
                cv2.namedWindow(painel.titulo, cv2.WINDOW_NORMAL)
                cv2.imshow(painel.titulo, img)

            if (cv2.waitKey(_REFRESH_MS) & 0xFF) == ord("q"):
                break
    finally:
        for painel in _PAINEIS:  # fecha todas, inclusive as desligadas no meio
            try:
                cv2.destroyWindow(painel.titulo)
            except cv2.error:
                pass


if __name__ == "__main__":
    print("=== VISUALIZADOR DE MASKS ===")
    print(
        "Janelas ligadas:", ", ".join(p.chave for p in painels_ativos()) or "(nenhuma)"
    )
    print("'q' para sair\n")
    run_masks_window()
