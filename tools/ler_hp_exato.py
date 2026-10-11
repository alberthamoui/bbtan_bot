"""
ler_hp_exato.py — HP dos blocos de uma captura REAL sem os erros do OCR do
bot, pra validar o jogo local com a entrada certa.

Análise por síntese: o texto do HP é escrito com a fonte do próprio jogo
(glifos hp_N extraídos das capturas, jogo_local/assets) numa posição que já
sabemos (geometria.py). Pra cada bloco, escreve cada HP candidato no mesmo
lugar e fica com o que mais se parece com a "tinta" real (quanto o pixel
escureceu em relação à cor do bloco).

    hp = ler_hp(board, coluna, fileira, tipo, divisor=False, maximo=300)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from jogo_local import fisica as fis
from jogo_local import geometria as geo
from jogo_local.desenho import Desenhista

_des = None


def _desenhista() -> Desenhista:
    global _des
    if _des is None:
        _des = Desenhista()
    return _des


def _texto(tipo: str, coluna: int, fileira: int, divisor: bool):
    """(centro x, topo) do texto do HP no tabuleiro — igual ao desenho.py."""
    x0, y0 = geo.x_coluna(coluna), geo.y_fileira(fileira)
    if divisor:
        x0, y0 = x0 + geo.DIV_CORPO_DESLOC[0], y0 + geo.DIV_CORPO_DESLOC[1]
    lado = geo.LADO_2X2 if tipo == "2x2" else geo.LADO
    if tipo.startswith("T"):
        return x0 + geo.TRI_TEXTO_CX[tipo[2]], y0 + geo.TRI_TEXTO_Y[tipo[1]]
    return x0 + lado / 2, y0 + lado / 2 + geo.HP_DY_CENTRO - geo.HP_ALTURA / 2


_meios = {}


def _glifo(c: str, fx: float, fy: float) -> np.ndarray:
    """Glifo do HP deslocado de (fx, fy) < 1 px (interpolado)."""
    k = (c, fx, fy)
    if k not in _meios:
        g = _desenhista().g_hp[c].astype(np.float32)
        g = np.pad(g, 1)
        m = np.float32([[1, 0, fx], [0, 1, fy]])
        _meios[k] = cv2.warpAffine(g, m, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
    return _meios[k]


def _por(out: np.ndarray, a: np.ndarray, gx: int, gy: int) -> None:
    h, w = out.shape
    ya, xa = max(0, gy), max(0, gx)
    yb, xb = min(h, gy + a.shape[0]), min(w, gx + a.shape[1])
    if yb > ya and xb > xa:
        out[ya:yb, xa:xb] = np.maximum(out[ya:yb, xa:xb], a[ya - gy : yb - gy, xa - gx : xb - gx])


def _escrever(txt: str, xc: float, topo: float, x0: int, y0: int, w: int, h: int,
              tinta: np.ndarray = None, dentro: np.ndarray = None):
    """Alfa do texto num recorte (x0, y0, w, h) do tabuleiro, como o desenho.py
    põe. Com `tinta`, cada dígito pode andar até 2 px pro lado (o jogo aperta
    alguns pares, ex: "71") e devolve (alfa, erro)."""
    d = _desenhista()
    out = np.zeros((h, w), np.float32)
    gs = [d.g_hp[c] for c in txt]
    larg = sum(g.shape[1] - 2 for g in gs) + d.m["hp_vao"] * (len(gs) - 1)
    x = xc - larg / 2
    gy = int(round(topo)) - 1 - y0
    for g in gs:
        gx = int(round(x)) - 1 - x0
        melhor = (None, 0)
        for dx in ((-2, -1, 0, 1, 2) if tinta is not None else (0,)):
            t = out.copy()
            _por(t, g, gx + dx, gy)
            e = float(np.abs(t - tinta)[dentro].sum()) if tinta is not None else 0.0
            if melhor[0] is None or e < melhor[0][0]:
                melhor = ((e, t), dx)
        out = melhor[0][1]
        x += g.shape[1] - 2 + d.m["hp_vao"] + melhor[1]
    if tinta is None:
        return out
    return out, float(np.abs(out - tinta)[dentro].sum())


def ler_hp(board: np.ndarray, coluna: int, fileira: int, tipo: str, divisor: bool = False,
           maximo: int = 300) -> int:
    xc, topo = _texto(tipo, coluna, fileira, divisor)
    x0, y0 = int(xc) - 32, int(topo) - 4
    w, h = 64, 24
    x0, y0 = max(0, x0), max(0, y0)
    rec = board[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
    h, w = rec.shape[:2]
    # só dentro do bloco (o fundo e o vão também são escuros, como o texto)
    dentro = np.zeros((h, w), np.uint8)
    poli = fis.vertices_bloco(coluna, fileira, tipo, encolher=1.5)
    if divisor:
        poli = [(x + geo.DIV_CORPO_DESLOC[0], y + geo.DIV_CORPO_DESLOC[1]) for x, y in poli]
    cv2.fillPoly(dentro, [np.round(np.array([(x - x0, y - y0) for x, y in poli]) * 16).astype(np.int32)], 1, shift=4)
    if divisor:  # o mini bloco e o contorno dele no canto também não são texto
        mx0, my1 = geo.x_coluna(coluna) + geo.DIV_MINI[0] - 3, geo.y_fileira(fileira) + geo.DIV_MINI[3] + 3
        dentro[: max(0, int(my1 - y0)), max(0, int(mx0 - x0)) :] = 0
    dentro = dentro.astype(bool)
    # cor do bloco = os pixels mais claros (o texto é preto); tinta = quanto
    # o pixel foi puxado da cor do bloco pro preto
    brilho = rec.sum(axis=2)
    cor = np.median(rec[dentro & (brilho >= np.percentile(brilho[dentro], 60))], axis=0)
    tinta = np.clip(1.0 - (rec @ cor) / max(float(cor @ cor), 1.0), 0, 1)
    notas = []
    for hp in range(1, maximo + 1):
        # o texto às vezes fica 1 px acima/abaixo do previsto (nos triângulos)
        e = min(_escrever(str(hp), xc, topo + dy, x0, y0, w, h, tinta, dentro)[1] for dy in (-1, 0, 1))
        notas.append((e, hp))
    # 2a passada nos 5 melhores: meio pixel pra cada lado e erro suavizado
    # (o 6 e o 5 só diferem num traço; meio pixel de desalinho decidia)
    suave = cv2.GaussianBlur(tinta, (3, 3), 0.8)
    melhor = None
    for _, hp in sorted(notas)[:5]:
        e = _refinar(str(hp), xc, topo, x0, y0, w, h, suave, dentro)
        if melhor is None or e < melhor[0]:
            melhor = (e, hp)
    return melhor[1]


def _refinar(txt, xc, topo, x0, y0, w, h, suave, dentro) -> float:
    d = _desenhista()
    gs = [d.g_hp[c] for c in txt]
    larg = sum(g.shape[1] - 2 for g in gs) + d.m["hp_vao"] * (len(gs) - 1)
    melhor = np.inf
    for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
        out = np.zeros((h, w), np.float32)
        x = xc - larg / 2
        for c, g in zip(txt, gs):
            esc = None
            for dx in np.arange(-2.0, 2.01, 0.5):
                xi, yi = x - 2 + dx - x0, topo - 2 + dy - y0  # glifo com 1 px de margem + 1 do pad
                ix, iy = int(np.floor(xi)), int(np.floor(yi))
                t = out.copy()
                _por(t, _glifo(c, round(xi - ix, 2), round(yi - iy, 2)), ix, iy)
                e = float(((cv2.GaussianBlur(t, (3, 3), 0.8) - suave) ** 2)[dentro].sum())
                if esc is None or e < esc[0]:
                    esc = (e, t, dx)
            out = esc[1]
            x += g.shape[1] - 2 + d.m["hp_vao"] + esc[2]
        melhor = min(melhor, float(((cv2.GaussianBlur(out, (3, 3), 0.8) - suave) ** 2)[dentro].sum()))
    return melhor
