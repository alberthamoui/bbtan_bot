"""
fisica.py — Física das bolas do jogo local, escrita do zero a partir das
MEDIÇÕES do jogo real (tools/medir_rajadas.py), sem nada do physics.py do bot.

Medido nas rajadas gravadas (60 fps):
  - o jogo anda em ticks fixos de 50 Hz (1 quadro repetido a cada 6);
  - a bola anda PASSO = 13,308 px por tick, sempre (665 px/s);
  - as bolas saem uma a cada INTERVALO_TICKS = 3 ticks (39,9 px de fila);
  - o rebote é um espelho exato no instante do contato: a bola não perde
    o resto do passo (as retas antes/depois são simétricas na parede);
  - paredes: o CENTRO da bola reflete em x = 11,0 / 439,3 e y = 12,7;
  - a bola pousa quando o centro desce até y = CHAO_Y, no x do cruzamento.

Colisão com blocos: o centro da bola contra a forma do bloco (quadrado,
triângulo, 2x2 nas posições de geometria.py) encolhida e depois engordada
pelo raio da bola, com cantos arredondados (normal radial no vértice).

Validação contra o jogo real (19 rajadas; as 7 de teste, turnos 41-71, não
entraram em nenhuma medida):
  - tools/validar_fisica_local.py: até o acelerar, todas as bolas a <= 5 px
    das reais em 97% (treino) / 100% (teste) do tempo, contra 76% / 82% do
    physics.py do bot;
  - tools/validar_rodadas_local.py: no fim da rodada inteira, HP exato em
    78% / 84% dos blocos, contra 47% / 50% do bot.
Medidas: python tools/medir_rajadas.py --rebotes
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import config as cfg
from jogo_local import geometria as geo

TICK_HZ = 50
TICK_S = 1.0 / TICK_HZ
PASSO = 13.308 * cfg.ESCALA  # px por tick
INTERVALO_TICKS = 3  # ticks entre uma bola e a próxima
PAREDE_ESQ = 11.0 * cfg.ESCALA  # limites do CENTRO da bola
PAREDE_DIR = 439.3 * cfg.ESCALA
TETO = 12.7 * cfg.ESCALA
CHAO_Y = 566.8 * cfg.ESCALA  # centro da bola parada
# Contato com bloco: a forma VISÍVEL do bloco encolhida (ENCOLHER nas faces
# da esquerda/direita, ENCOLHER_Y nas de cima/baixo) e engordada pelo raio da
# bola. Na face, o centro para a RAIO_BOLA - encolhido da borda visível.
# Medido nos rebotes limpos: ~5,0 px nas faces laterais (5,14 / 5,02 / 5,15 /
# 4,5) e ~8 px nas de cima/baixo (6,8 / 8,5 / 9,2); o raio é o da bola na tela
# (área 260 px² -> 9,1-9,2). Antes, um raio único de 6 px pra tudo: mesma
# fidelidade no treino, mas erro p90 de 1,8 px contra 1,1 px.
# antigo: RAIO_BOLA = 6.0, ENCOLHER = 0.0, ENCOLHER_Y = None
RAIO_BOLA = 9.2 * cfg.ESCALA
ENCOLHER = 4.2 * cfg.ESCALA  # faces da esquerda/direita: contato a 5,0 px
ENCOLHER_Y = 1.2 * cfg.ESCALA  # faces de cima/baixo: contato a 8,0 px
# Hipotenusa do triângulo: continua a 45 graus (a reflexão real troca os
# componentes da velocidade exatamente) e o contato fica a 9,7 px da borda
# visível (rajada_j1_t001: entra (-1,34; -13,23), sai (-13,26; -1,34)).
# antigo: o triângulo saía da caixa encolhida em x e y diferentes, o que
# entortava a hipotenusa ~3 graus
ENCOLHER_HIP = -0.5 * cfg.ESCALA
# Distância centro-centro que pega o power-up. O ícone some 1-2 quadros
# DEPOIS de pego: na rajada_t031 a bola mais perto antes do sumiço passou a
# 17,9 px, então o raio é >= 18; com 18-22 o jogo local acerta quantas bolas
# a rodada ganhou em 12 de 17 rodadas (com 14, 11). Estimativa.
# antigo: RAIO_POWERUP = 22.0 / 14.0
RAIO_POWERUP = 20.0 * cfg.ESCALA
# Tela limpa (sem blocos): as bolas voando voltam em linha reta pro ponto de
# partida, a 19,55 px por quadro de 60 fps (1173 px/s) — rajada_t001
PASSO_RECOLHER = 1173.0 / TICK_HZ * cfg.ESCALA
# Acelerar: 4 ticks por quadro de 60 fps (53,2 px) = 240 ticks/s; o jogo
# demora ~0,12 s pra responder ao clique (rajada_t001)
TICKS_POR_S_ACELERADO = 240
ATRASO_ACELERAR_S = 0.12
# Do fim do arrasto (mouse solto) até a 1a bola sair: mediana de 19 rajadas
ATRASO_DISPARO_S = 0.045
MAX_REBOTES_POR_TICK = 8

Vetor = Tuple[float, float]


# ── formas ─────────────────────────────────────────────────────────────────
def vertices_bloco(coluna: int, fileira: int, tipo: str, encolher: float = None) -> List[Vetor]:
    """Polígono (horário na tela) do bloco, em px do tabuleiro: a parte
    VISÍVEL com cada aresta recuada pra dentro — ENCOLHER nas verticais,
    ENCOLHER_Y nas horizontais, ENCOLHER_HIP na hipotenusa. Com `encolher`,
    recua tudo igual (ex: máscara do texto)."""
    x0, y0 = geo.x_coluna(coluna), geo.y_fileira(fileira)
    lado = geo.LADO_2X2 if tipo == "2x2" else geo.LADO
    x1, y1 = x0 + lado, y0 + lado
    tl, tr, br, bl = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
    if tipo in ("Q", "2x2"):
        v = [tl, tr, br, bl]
    else:  # triângulo: a letra é o canto do ângulo reto; some o canto oposto
        v = {"Ttl": [tl, tr, bl], "Ttr": [tl, tr, br], "Tbr": [tr, br, bl], "Tbl": [tl, br, bl]}[tipo]
    if encolher is not None:
        return _encolher_poligono(v, lambda ex, ey: encolher)
    ey_ = ENCOLHER if ENCOLHER_Y is None else ENCOLHER_Y

    def recuo(ex, ey):
        if abs(ey) < 1e-9:
            return ey_  # horizontal
        if abs(ex) < 1e-9:
            return ENCOLHER  # vertical
        return ENCOLHER_HIP

    return _encolher_poligono(v, recuo)


def _encolher_poligono(v: List[Vetor], e) -> List[Vetor]:
    """Desloca cada aresta de um polígono convexo pra dentro: `e(ex, ey)` px,
    pela direção (ex, ey) da aresta (ou `e` fixo)."""
    if not callable(e):
        fixo = e
        e = lambda ex, ey: fixo  # noqa: E731
    n = len(v)
    area = sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))
    sinal = 1.0 if area > 0 else -1.0
    linhas = []
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        L = math.hypot(ex, ey)
        nx, ny = sinal * ey / L, -sinal * ex / L  # pra fora
        k = e(ex, ey)
        linhas.append(((a[0] - k * nx, a[1] - k * ny), (ex, ey)))
    out = []
    for i in range(n):
        (p, d), (q, f) = linhas[i - 1], linhas[i]
        den = d[0] * f[1] - d[1] * f[0]
        t = ((q[0] - p[0]) * f[1] - (q[1] - p[1]) * f[0]) / den
        out.append((p[0] + t * d[0], p[1] + t * d[1]))
    return out


@dataclass
class Forma:
    """Polígono convexo engordado por `raio` (soma de Minkowski com um disco)."""

    vertices: List[Vetor]
    raio: float
    dono: object = None  # o bloco (ou None pra paredes)
    arestas: list = field(default_factory=list)  # (a, b, normal pra fora)

    def __post_init__(self):
        v = self.vertices
        n = len(v)
        # área com sinal pra saber o sentido; normal pra fora de cada aresta
        area = sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))
        sinal = 1.0 if area > 0 else -1.0
        for i in range(n):
            a, b = v[i], v[(i + 1) % n]
            ex, ey = b[0] - a[0], b[1] - a[1]
            L = math.hypot(ex, ey)
            nx, ny = sinal * ey / L, -sinal * ex / L
            self.arestas.append((a, b, (nx, ny)))

    def entrada(self, p: Vetor, d: Vetor, alcance: float) -> Optional[Tuple[float, Vetor]]:
        """Primeiro contato do centro (p + t·d, |d| = 1) com a forma engordada,
        0 <= t <= alcance. Devolve (t, normal) ou None."""
        melhor = None
        r = self.raio
        for a, b, (nx, ny) in self.arestas:
            dn = d[0] * nx + d[1] * ny
            if dn >= -1e-9:
                continue  # indo pra longe (ou paralelo) desta face
            # reta da face deslocada de r pra fora
            dist = (p[0] - a[0]) * nx + (p[1] - a[1]) * ny - r
            t = -dist / dn
            if t < -1e-6 or t > alcance or (melhor and t >= melhor[0]):
                continue
            qx, qy = p[0] + t * d[0] - r * nx, p[1] + t * d[1] - r * ny
            ex, ey = b[0] - a[0], b[1] - a[1]
            u = ((qx - a[0]) * ex + (qy - a[1]) * ey) / (ex * ex + ey * ey)
            if -1e-9 <= u <= 1 + 1e-9:
                melhor = (max(t, 0.0), (nx, ny))
        if r > 0:
            for cx, cy in self.vertices:
                fx, fy = p[0] - cx, p[1] - cy
                bq = fx * d[0] + fy * d[1]
                c = fx * fx + fy * fy - r * r
                if bq >= 0:
                    continue  # se afastando do vértice
                disc = bq * bq - c
                if disc < 0:
                    continue
                t = -bq - math.sqrt(disc)
                if t < -1e-6 or t > alcance or (melhor and t >= melhor[0]):
                    continue
                t = max(t, 0.0)
                hx, hy = p[0] + t * d[0] - cx, p[1] + t * d[1] - cy
                L = math.hypot(hx, hy) or 1.0
                melhor = (t, (hx / L, hy / L))
        return melhor


def refletir(d: Vetor, n: Vetor) -> Vetor:
    k = 2 * (d[0] * n[0] + d[1] * n[1])
    x, y = d[0] - k * n[0], d[1] - k * n[1]
    L = math.hypot(x, y)
    return (x / L, y / L)


# ── a bola ─────────────────────────────────────────────────────────────────
@dataclass
class Bola:
    x: float
    y: float
    dx: float  # direção (unitária)
    dy: float
    voando: bool = True
    pousou_em: Optional[float] = None
    # power-ups que esta bola já pegou (laser/R valem uma vez por passagem)
    dentro_de: set = field(default_factory=set)


def direcao(angulo_graus: float) -> Vetor:
    """0 = direita, 90 = pra cima (convenção do bot)."""
    a = math.radians(angulo_graus)
    return (math.cos(a), -math.sin(a))


@dataclass
class Contato:
    """O que aconteceu num tick (pra lógica do jogo aplicar dano/power-ups)."""

    bola: Bola
    forma: Optional[Forma]  # None = parede/teto
    x: float
    y: float


def avancar(bola: Bola, formas: List[Forma], distancia: float = PASSO,
            ao_bater: Optional[Callable[[Contato], bool]] = None) -> List[Contato]:
    """
    Anda `distancia` com rebote exato. `ao_bater(contato)` é chamado em cada
    batida em bloco; se devolver False, a forma some na hora (bloco quebrou)
    mas a bola reflete do mesmo jeito.
    """
    contatos: List[Contato] = []
    resto = distancia
    for _ in range(MAX_REBOTES_POR_TICK):
        if not bola.voando or resto <= 1e-9:
            break
        p, d = (bola.x, bola.y), (bola.dx, bola.dy)
        melhor_t, melhor_n, melhor_f = resto, None, None
        # paredes e teto (o centro tem limites fixos)
        if d[0] < 0:
            t = (PAREDE_ESQ - p[0]) / d[0]
            if 0 <= t < melhor_t:
                melhor_t, melhor_n, melhor_f = t, (1.0, 0.0), None
        elif d[0] > 0:
            t = (PAREDE_DIR - p[0]) / d[0]
            if 0 <= t < melhor_t:
                melhor_t, melhor_n, melhor_f = t, (-1.0, 0.0), None
        if d[1] < 0:
            t = (TETO - p[1]) / d[1]
            if 0 <= t < melhor_t:
                melhor_t, melhor_n, melhor_f = t, (0.0, 1.0), None
        for f in formas:
            e = f.entrada(p, d, melhor_t)
            if e is not None and e[0] < melhor_t:
                melhor_t, (melhor_n), melhor_f = e[0], e[1], f
        # chão: pousa no cruzamento (antes de qualquer outra batida mais longe)
        if d[1] > 0:
            t = (CHAO_Y - p[1]) / d[1]
            if 0 <= t <= melhor_t and (melhor_n is None or t < melhor_t):
                bola.x, bola.y = p[0] + t * d[0], CHAO_Y
                bola.voando = False
                bola.pousou_em = bola.x
                break
        bola.x, bola.y = p[0] + melhor_t * d[0], p[1] + melhor_t * d[1]
        resto -= melhor_t
        if melhor_n is None:
            break
        bola.dx, bola.dy = refletir(d, melhor_n)
        if melhor_f is not None:
            c = Contato(bola, melhor_f, bola.x, bola.y)
            contatos.append(c)
            if ao_bater is not None and ao_bater(c) is False and melhor_f in formas:
                formas.remove(melhor_f)
        # evita grudar: se a direção nova ainda entra na forma, o próximo laço acha t = 0
    return contatos


def angulo_aleatorio_pra_cima(rng: random.Random) -> Vetor:
    """Power-up R: direção aleatória sempre pra cima (faixa a medir)."""
    return direcao(rng.uniform(15.0, 165.0))
