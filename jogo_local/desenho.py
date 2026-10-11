"""
desenho.py — Transforma um Estado na imagem da JANELA do jogo (a mesma área
que o modo --referencia captura: o tabuleiro com MARGEM_X/MARGEM_CIMA em
volta), pixel a pixel, usando os recortes reais de jogo_local/assets/
(tools/extrair_assets_jogo.py).

  img = Desenhista().desenhar(estado)   # BGR uint8 (altura, largura, 3)

Bordas de blocos e triângulos têm antialiasing por cobertura (supersampling
4x), como o jogo, que desenha em posições fracionárias (passo de 63,5 px).
Os textos usam os glifos reais (alfa médio de dezenas de ocorrências).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from jogo_local import geometria as geo
from jogo_local.estado import Bloco, Estado

PASTA_ASSETS = Path(__file__).resolve().parent / "assets"
SS = 4  # supersampling das bordas
SOMBRA = (0.8, 0.75)  # deslocamento (px) da sombra preta do mini bloco do divisor
CONTORNO = 0.9  # px de contorno preto em volta dos blocos (ajustado nos vãos entre blocos)
# "sombra" (forma deslocada SOMBRA) ou "contorno" (forma ampliada CONTORNO).
# Medido em 71 capturas: sombra 65/71 blocos lidos iguais pela visão do bot e
# diferença média 0,99/255; contorno 64/71 e 1,05.
BORDA_ESTILO = "sombra"


def _ampliar(pontos, d: float):
    """Polígono convexo ampliado d px pra fora (desloca cada lado ao longo da sua normal)."""
    p = np.array(pontos, np.float64)
    n = len(p)
    area = 0.5 * sum(p[i, 0] * p[(i + 1) % n, 1] - p[(i + 1) % n, 0] * p[i, 1] for i in range(n))
    sinal = 1.0 if area > 0 else -1.0
    linhas = []
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        t = (b - a) / np.linalg.norm(b - a)
        normal = sinal * np.array([t[1], -t[0]])  # aponta pra fora
        linhas.append((a + normal * d, t))
    out = []
    for i in range(n):
        (p1, t1), (p2, t2) = linhas[i - 1], linhas[i]
        m = np.array([t1, -t2]).T
        s = np.linalg.solve(m, p2 - p1)
        out.append(tuple(p1 + t1 * s[0]))
    return out

# Degradê da cor do bloco em função de r = (HP - 1) / (base - 1), base >= 6.
# Medido nas capturas: r = 1 roxo (bloco cheio), caindo pra amarelo conforme
# perde HP; r = 2 verde (bloco de HP dobrado cheio). Pontos em BGR.
DEGRADE = [
    (0.0, (33, 225, 255)),
    (0.2, (33, 173, 255)),
    (0.4, (33, 111, 255)),
    (0.6, (33, 59, 255)),
    (0.8, (96, 33, 255)),
    (1.0, (255, 0, 137)),
    (2.0, (116, 255, 31)),
]


def cor_bloco(hp: int, hp_base: int):
    """BGR do bloco: posição no DEGRADE por r = (hp - 1) / (max(base, 6) - 1)."""
    base = max(hp_base, 6)
    r = (hp - 1) / (base - 1)
    r = min(max(r, DEGRADE[0][0]), DEGRADE[-1][0])
    for (r0, c0), (r1, c1) in zip(DEGRADE, DEGRADE[1:]):
        if r <= r1:
            t = (r - r0) / (r1 - r0)
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))
    return DEGRADE[-1][1]


def _ler_alfa(p: Path) -> np.ndarray:
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def _ler_rgba(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return im[..., :3].astype(np.float32), im[..., 3].astype(np.float32) / 255.0


class Desenhista:
    def __init__(self, pasta: Path = PASTA_ASSETS):
        self.m = json.load(open(pasta / "assets.json", encoding="utf-8"))
        self.fundo = cv2.imread(str(pasta / "fundo.png"))
        self.barra_rodada = cv2.imread(str(pasta / "barra_rodada.png"))
        self.robo = _ler_rgba(pasta / "robo.png")
        self.bola = _ler_rgba(pasta / "bola.png")
        self.pu: Dict[str, tuple] = {}
        for tipo, nome in (("+", "mais"), ("LH", "laser_h"), ("LV", "laser_v"), ("R", "aleatorio"), ("B", "moeda")):
            p = pasta / f"pu_{nome}.png"
            if p.exists():
                self.pu[tipo] = _ler_rgba(p)
        self.g_hp = {d: _ler_alfa(pasta / f"hp_{d}.png") for d in self.m["glifos_hp"]}
        self.g_linha = {d: _ler_alfa(pasta / f"linha_{d}.png") for d in self.m["glifos_linha"]}
        self.g_cont = {d: _ler_alfa(pasta / f"contador_{d}.png") for d in self.m["glifos_contador"]}
        self.icone = {n: _ler_alfa(pasta / f"icone_{n}.png") for n in self.m["icones"]}
        self.bx = self.m["janela"]["tabuleiro_x"]
        self.by = self.m["janela"]["tabuleiro_y"]
        self.cor_fundo = np.array(self.m["cor_fundo_bgr"], np.float32)

    # ── primitivas ──────────────────────────────────────────────────────────
    @staticmethod
    def _misturar(img: np.ndarray, x: int, y: int, cor, alfa: np.ndarray) -> None:
        """img[y:, x:] = cor*alfa + img*(1-alfa), recortando nas bordas. cor: BGR ou imagem (h, w, 3)."""
        h, w = alfa.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return
        a = alfa[y0 - y : y1 - y, x0 - x : x1 - x][..., None]
        c = cor[y0 - y : y1 - y, x0 - x : x1 - x] if isinstance(cor, np.ndarray) and cor.ndim == 3 else np.array(cor, np.float32)
        reg = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(c * a + reg * (1 - a), 0, 255).astype(np.uint8)

    @staticmethod
    def _cobertura(pontos, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """Alfa (h, w) da área do polígono (coordenadas absolutas float), por supersampling."""
        m = np.zeros((h * SS, w * SS), np.uint8)
        pts = np.array([[(px - x0) * SS, (py - y0) * SS] for px, py in pontos], np.float32)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def _texto(self, img, glifos, txt: str, x_centro: float, y_topo: float, vao: float, cor) -> None:
        """Escreve txt com os glifos (recortes com 1 px de margem), centrado em x, topo do traço em y."""
        gs = [glifos[c] for c in txt if c in glifos]
        if not gs:
            return
        larg = sum(g.shape[1] - 2 for g in gs) + vao * (len(gs) - 1)
        x = x_centro - larg / 2
        for g in gs:
            self._misturar(img, int(round(x)) - 1, int(round(y_topo)) - 1, cor, g)
            x += g.shape[1] - 2 + vao

    # ── elementos ───────────────────────────────────────────────────────────
    def _bloco(self, img, b: Bloco) -> None:
        bx, by = self.bx, self.by
        cx0 = bx + geo.x_coluna(b.coluna)  # canto da célula
        cy0 = by + geo.y_fileira(b.fileira)
        x0, y0 = cx0, cy0
        if b.divisor:  # o corpo do divisor fica deslocado (medido)
            x0, y0 = cx0 + geo.DIV_CORPO_DESLOC[0], cy0 + geo.DIV_CORPO_DESLOC[1]
        if b.tipo == "2x2":
            lx = geo.x_coluna(b.coluna + 1) + geo.largura_coluna(b.coluna + 1) - geo.x_coluna(b.coluna)
            ly = geo.y_fileira(b.fileira + 1) + geo.altura_fileira(b.fileira + 1) - geo.y_fileira(b.fileira)
        else:
            lx, ly = geo.largura_coluna(b.coluna), geo.altura_fileira(b.fileira)
        lado = lx
        cor = b.cor if b.cor is not None else cor_bloco(b.hp, b.hp_base)
        tl, tr, bl, br = (x0, y0), (x0 + lx, y0), (x0, y0 + ly), (x0 + lx, y0 + ly)
        if b.tipo.startswith("T"):
            canto = b.tipo[1:]  # tl, tr, bl, br: canto do ângulo reto
            pontos = {"tl": [tl, tr, bl], "tr": [tl, tr, br], "bl": [tl, bl, br], "br": [tr, br, bl]}[canto]
        else:
            pontos = [tl, tr, br, bl]
        ix, iy = int(np.floor(x0)), int(np.floor(y0))
        w, h = int(np.ceil(lx)) + 3, int(np.ceil(ly)) + 3
        # contorno preto: a mesma forma ampliada CONTORNO px pra todos os lados,
        # por baixo do preenchimento (ajustado nos perfis dos vãos entre blocos
        # vizinhos: pixel escuro dos dois lados do vão).
        # antigo: sombra = mesma forma deslocada SOMBRA pra baixo/direita
        ix, iy = ix - 2, iy - 2
        w, h = w + 3, h + 3
        if BORDA_ESTILO == "contorno":
            borda = _ampliar(pontos, CONTORNO)
        else:
            borda = [(px + SOMBRA[0], py + SOMBRA[1]) for px, py in pontos]
        self._misturar(img, ix, iy, (0, 0, 0), self._cobertura(borda, ix, iy, w, h))
        self._misturar(img, ix, iy, cor, self._cobertura(pontos, ix, iy, w, h))

        # ícones no canto superior direito (antes do número, como no jogo)
        k, fora = self.m["icone_lado"], self.m["icone_fora"]
        xi, yi = int(round(x0 + lado)) + fora - k, int(round(y0)) - fora
        if b.divisor:
            # antigo: recorte médio (icone_divisor*.png) — a mediana de posições
            # fracionárias diferentes deixava o L 1 px baixo e ele encostava no
            # número (o OCR lia 44 como 47). Agora desenhado com a geometria medida.
            mx0, my0, mx1, my1 = (cx0 + geo.DIV_MINI[0], cy0 + geo.DIV_MINI[1], cx0 + geo.DIV_MINI[2], cy0 + geo.DIV_MINI[3])
            mini = [(mx0, my0), (mx1, my0), (mx1, my1), (mx0, my1)]
            ix2, iy2 = int(np.floor(mx0)) - 1, int(np.floor(my0)) - 1
            w2, h2 = int(np.ceil(mx1 - mx0)) + 4, int(np.ceil(my1 - my0)) + 4
            sombra2 = [(px + SOMBRA[0], py + SOMBRA[1]) for px, py in mini]
            self._misturar(img, ix2, iy2, (0, 0, 0), self._cobertura(sombra2, ix2, iy2, w2, h2))
            self._misturar(img, ix2, iy2, cor, self._cobertura(mini, ix2, iy2, w2, h2))
            e = geo.DIV_CONTORNO
            for poli in (
                [(mx0, y0), (mx0 + e, y0), (mx0 + e, my1), (mx0, my1)],  # lado esquerdo, só sobre o corpo
                [(mx0, my1 - e), (mx1 - e, my1 - e), (mx1 - e, my1), (mx0, my1)],  # base
            ):
                self._misturar(img, ix2, iy2, (0, 0, 0), self._cobertura(poli, ix2, iy2, w2, h2))
        if b.corrente is not None and "corrente" in self.icone:
            self._misturar(img, xi, yi, (0, 0, 0), self.icone["corrente"])

        txt = str(b.hp)
        if b.tipo.startswith("T"):
            cx = x0 + geo.TRI_TEXTO_CX[b.tipo[2]]
            topo = y0 + geo.TRI_TEXTO_Y[b.tipo[1]]
        else:
            cx = x0 + lx / 2
            topo = y0 + ly / 2 + geo.HP_DY_CENTRO - geo.HP_ALTURA / 2
        self._texto(img, self.g_hp, txt, cx, topo, self.m["hp_vao"], (0, 0, 0))

    def _powerup(self, img, p) -> None:
        if p.tipo not in self.pu:
            return
        cor, a = self.pu[p.tipo]
        cx, cy = geo.centro_celula(p.coluna, p.fileira)
        r = a.shape[0] // 2
        self._misturar(img, int(round(self.bx + cx)) - r, int(round(self.by + cy)) - r, cor, a)

    def _robo(self, img, est: Estado) -> None:
        cor, a = self.robo
        x = est.robo_x if est.robo_x is not None else est.bola_x
        ox, oy = self.m["robo_offset"]
        self._misturar(img, int(round(self.bx + x)) + ox, int(round(self.by + geo.BOLA_Y)) + oy, cor, a)
        if not est.em_rodada:
            dx, dy = self.m["contador_offset"]
            txt = "x" + str(est.bolas)
            # o offset é do canto do "x"; _texto centraliza, então converte
            gs = [self.g_cont[c] for c in txt if c in self.g_cont]
            larg = sum(g.shape[1] - 2 for g in gs) + self.m["contador_vao"] * (len(gs) - 1)
            x_txt = self.bx + est.bola_x + dx + larg / 2
            # topo: o offset vertical é dos dígitos; o "x" é mais baixo, alinhado pela base
            y_txt = self.by + geo.BOLA_Y + dy
            self._texto_contador(img, txt, x_txt, y_txt)

    def _texto_contador(self, img, txt, x_centro, y_topo_digitos) -> None:
        gs = [(c, self.g_cont[c]) for c in txt if c in self.g_cont]
        larg = sum(g.shape[1] - 2 for _, g in gs) + self.m["contador_vao"] * (len(gs) - 1)
        base = y_topo_digitos + (self.g_cont["0"].shape[0] - 2)  # linha de base dos dígitos
        x = x_centro - larg / 2
        for c, g in gs:
            topo = base - (g.shape[0] - 2)
            self._misturar(img, int(round(x)) - 1, int(round(topo)) - 1, (255, 255, 255), g)
            x += g.shape[1] - 2 + self.m["contador_vao"]

    # ── tela inteira ────────────────────────────────────────────────────────
    def desenhar(self, est: Estado) -> np.ndarray:
        img = self.fundo.copy()
        if est.em_rodada:
            y = self.m["y_barra_rodada"]
            img[y : y + self.barra_rodada.shape[0]] = self.barra_rodada
        self._texto(
            img, self.g_linha, str(est.rodada),
            self.bx + self.m["linha_centro_x"], self.by + self.m["linha_topo_y"], 7.0, (255, 255, 255),
        )
        for b in est.blocos:
            self._bloco(img, b)
        for p in est.powerups:
            self._powerup(img, p)
        if est.mostrar_robo:
            self._robo(img, est)
        cor, a = self.bola
        ox, oy = self.m["bola_offset"]
        for x, y in est.bolas_voando:
            self._misturar(img, int(round(self.bx + x)) + ox, int(round(self.by + y)) + oy, cor, a)
        return img
