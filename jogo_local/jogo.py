"""
jogo.py — Regras do jogo local: uma rodada tick a tick (50 Hz) e a passagem
de uma rodada pra outra.

Rodada(estado, angulo) solta as bolas da posição da bola parada, uma a cada
fisica.INTERVALO_TICKS, e aplica as regras contadas/medidas no jogo real:
  - cada batida tira 1 de HP; HP 0 quebra o bloco (a bola reflete mesmo assim);
  - corrente ("Share Block"): o par divide o mesmo HP;
  - divisor: ao quebrar, nascem 2 blocos com metade do HP em células vazias
    sorteadas;
  - "+": some na 1a bola, +1 bola na próxima rodada; "B": some, só pontos;
  - LH / LV: cada bola que passa tira 1 de HP de toda a fileira / coluna;
    somem no fim da rodada se alguma bola passou;
  - R: cada bola que passa sai numa direção aleatória pra cima; some no fim
    da rodada se alguma bola passou;
  - a 1a bola que pousa decide onde a próxima rodada começa;
  - sem blocos na tela (ou retrieve), as bolas voando voltam em linha reta
    pro ponto de partida e a rodada acaba, com o dano já feito.

Jogo junta as rodadas: depois de cada uma, tudo desce uma fileira, nasce a
fileira nova (gerador.py, com as estatísticas das capturas reais) e, se um
bloco passaria da última fileira, é game over.
"""

import copy
import math
import random
from typing import Dict, List, Optional

from jogo_local import fisica as fis
from jogo_local import geometria as geo
from jogo_local.estado import Bloco, Estado, PowerUp
from jogo_local.gerador import Gerador

CENTRO_X = (fis.PAREDE_ESQ + fis.PAREDE_DIR) / 2  # 1a rodada: bola no meio (225,1; medido 225,06)
ULTIMA_FILEIRA = geo.FILEIRAS - 1  # L7


def centro_powerup(p: PowerUp):
    return geo.centro_celula(p.coluna, p.fileira)


def celulas_de(b: Bloco):
    if b.tipo == "2x2":
        return [(b.coluna + i, b.fileira + j) for i in (0, 1) for j in (0, 1)]
    return [(b.coluna, b.fileira)]


class Rodada:
    def __init__(self, estado: Estado, angulo: float, rng: Optional[random.Random] = None):
        self.est = estado
        self.rng = rng or random.Random()
        self.angulo = angulo
        self.partida_x = estado.bola_x
        self.tick = 0
        self.lancadas = 0
        self.bolas: List[fis.Bola] = []
        self.primeira_pousou: Optional[float] = None
        self.bolas_ganhas = 0
        self.moedas = 0
        self.acabou = False
        self.recolhendo: Optional[float] = None  # x pra onde as bolas estão voltando
        self.pus_usados: set = set()  # ids de LH/LV/R que alguma bola tocou
        self.formas: List[fis.Forma] = []
        self._forma_de: Dict[int, fis.Forma] = {}
        for b in estado.blocos:
            self._criar_forma(b)

    # ── blocos ──────────────────────────────────────────────────────────────
    def _criar_forma(self, b: Bloco) -> None:
        f = fis.Forma(fis.vertices_bloco(b.coluna, b.fileira, b.tipo), fis.RAIO_BOLA, dono=b)
        self.formas.append(f)
        self._forma_de[id(b)] = f

    def _remover_bloco(self, b: Bloco) -> None:
        if b in self.est.blocos:
            self.est.blocos.remove(b)
        f = self._forma_de.pop(id(b), None)
        if f is not None and f in self.formas:
            self.formas.remove(f)

    def _dano(self, b: Bloco, n: int = 1) -> None:
        if b not in self.est.blocos:
            return
        par = [o for o in self.est.blocos if b.corrente is not None and o.corrente == b.corrente]
        for o in par or [b]:
            o.hp -= n
        if b.hp <= 0:
            for o in par or [b]:
                self._quebrar(o)

    def _quebrar(self, b: Bloco) -> None:
        self._remover_bloco(b)
        if b.divisor:
            metade = max(1, b.hp_base // 2)
            livres = self._celulas_livres()
            for c, f in self.rng.sample(livres, min(2, len(livres))):
                filho = Bloco(coluna=c, fileira=f, tipo="Q", hp=metade, hp_base=b.hp_base)
                self.est.blocos.append(filho)
                self._criar_forma(filho)
        if not self.est.blocos:
            self._recolher()

    def _celulas_livres(self):
        """Células vazias onde já existe jogo (L1-L6), sem bola dentro."""
        ocupadas = {c for b in self.est.blocos for c in celulas_de(b)}
        ocupadas |= {(p.coluna, p.fileira) for p in self.est.powerups}
        livres = []
        for f in range(1, ULTIMA_FILEIRA):
            for c in range(geo.COLUNAS):
                if (c, f) in ocupadas:
                    continue
                cx, cy = geo.centro_celula(c, f)
                if any(b.voando and abs(b.x - cx) < 45 and abs(b.y - cy) < 45 for b in self.bolas):
                    continue
                livres.append((c, f))
        return livres

    # ── power-ups ───────────────────────────────────────────────────────────
    def _powerups(self, bola: fis.Bola, x0: float, y0: float) -> None:
        """Power-ups que o centro da bola tocou entre (x0, y0) e a posição atual."""
        for p in list(self.est.powerups):
            cx, cy = centro_powerup(p)
            dentro = _dist_segmento(cx, cy, x0, y0, bola.x, bola.y) <= fis.RAIO_POWERUP
            chave = id(p)
            if not dentro:
                bola.dentro_de.discard(chave)
                continue
            if chave in bola.dentro_de:
                continue  # ainda passando pelo mesmo
            bola.dentro_de.add(chave)
            if p.tipo == "+":
                self.est.powerups.remove(p)
                self.bolas_ganhas += 1
            elif p.tipo == "B":
                self.est.powerups.remove(p)
                self.moedas += 1
            elif p.tipo in ("LH", "LV"):
                self.pus_usados.add(chave)
                alvo = [b for b in list(self.est.blocos)
                        if any((f == p.fileira) if p.tipo == "LH" else (c == p.coluna) for c, f in celulas_de(b))]
                for b in alvo:
                    self._dano(b)
                    if self.recolhendo is not None:
                        return
            elif p.tipo == "R":
                self.pus_usados.add(chave)
                bola.dx, bola.dy = fis.angulo_aleatorio_pra_cima(self.rng)

    # ── fim ─────────────────────────────────────────────────────────────────
    def _recolher(self) -> None:
        """Tela limpa ou retrieve: as que ainda não saíram ficam; as voando
        voltam em linha reta (fisica.PASSO_RECOLHER por tick)."""
        if self.recolhendo is None:
            self.recolhendo = self.primeira_pousou if self.primeira_pousou is not None else self.partida_x
            self.lancadas = self.est.bolas

    def retrieve(self) -> None:
        if not self.acabou:
            self._recolher()

    def _fechar(self) -> None:
        self.acabou = True
        self.est.powerups = [p for p in self.est.powerups if id(p) not in self.pus_usados]
        if self.recolhendo is not None:
            self.est.bola_x = self.recolhendo
        elif self.primeira_pousou is not None:
            self.est.bola_x = self.primeira_pousou

    # ── tick ────────────────────────────────────────────────────────────────
    def passo(self) -> None:
        """Avança 1 tick (1/50 s de jogo)."""
        if self.acabou:
            return
        if self.recolhendo is not None:
            self._passo_recolher()
            return
        if self.lancadas < self.est.bolas and self.tick == self.lancadas * fis.INTERVALO_TICKS:
            dx, dy = fis.direcao(self.angulo)
            self.bolas.append(fis.Bola(self.partida_x, fis.CHAO_Y, dx, dy))
            self.lancadas += 1
        for bola in self.bolas:
            if not bola.voando:
                continue
            x0, y0 = bola.x, bola.y
            fis.avancar(bola, self.formas, fis.PASSO, ao_bater=self._ao_bater)
            if self.recolhendo is not None:
                break
            self._powerups(bola, x0, y0)
            if self.recolhendo is not None:
                break
            if not bola.voando and self.primeira_pousou is None:
                self.primeira_pousou = bola.pousou_em
        self.tick += 1
        if self.lancadas >= self.est.bolas and not any(b.voando for b in self.bolas):
            self._fechar()

    def _passo_recolher(self) -> None:
        alvo = (self.recolhendo, fis.CHAO_Y)
        for b in self.bolas:
            if not b.voando:
                continue
            d = math.hypot(alvo[0] - b.x, alvo[1] - b.y)
            if d <= fis.PASSO_RECOLHER:
                b.x, b.y, b.voando = alvo[0], alvo[1], False
            else:
                b.x += (alvo[0] - b.x) / d * fis.PASSO_RECOLHER
                b.y += (alvo[1] - b.y) / d * fis.PASSO_RECOLHER
        self.tick += 1
        if not any(b.voando for b in self.bolas):
            self._fechar()

    def _ao_bater(self, c: fis.Contato) -> bool:
        b = c.forma.dono
        self._dano(b)
        return b in self.est.blocos

    def voando(self):
        return [(b.x, b.y) for b in self.bolas if b.voando]


def _dist_segmento(px, py, ax, ay, bx, by) -> float:
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / L2))
    return math.hypot(ax + u * ex - px, ay + u * ey - py)


class Jogo:
    """
    Uma partida: Jogo.disparar(angulo) começa a rodada, Jogo.passo() avança
    1 tick, Jogo.tela() devolve o Estado pra desenhar. Sem relógio: quem
    chama passo() decide o ritmo (tempo_real.py faz 50 ou 240 por segundo).
    """

    def __init__(self, seed: Optional[int] = None, gerador: Optional[Gerador] = None):
        self.rng = random.Random(seed)
        self.gerador = gerador or Gerador(self.rng)
        self.rodada: Optional[Rodada] = None
        self.partidas = 0
        self.novo_jogo()

    def novo_jogo(self) -> None:
        self.est = Estado(rodada=1, bola_x=CENTRO_X, bolas=1)
        self.gerador.nova_fileira(self.est)
        self.rodada = None
        self.game_over = False
        self.moedas = 0
        self.partidas += 1

    # ── ações ───────────────────────────────────────────────────────────────
    def disparar(self, angulo: float) -> bool:
        if self.rodada is not None or self.game_over:
            return False
        self.rodada = Rodada(self.est, angulo, self.rng)
        return True

    def retrieve(self) -> None:
        if self.rodada is not None:
            self.rodada.retrieve()

    def passo(self) -> None:
        if self.rodada is None:
            return
        self.rodada.passo()
        if self.rodada.acabou:
            self._proxima_rodada()

    def jogar_rodada(self, angulo: float, max_ticks: int = 50 * 120) -> int:
        """Sem relógio: dispara e roda até a rodada acabar. Devolve os ticks."""
        if not self.disparar(angulo):
            return 0
        n = 0
        while self.rodada is not None and n < max_ticks:
            self.passo()
            n += 1
        if self.rodada is not None:  # loop quase horizontal: retrieve, como o bot faz
            self.retrieve()
            while self.rodada is not None:
                self.passo()
                n += 1
        return n

    # ── entre rodadas ───────────────────────────────────────────────────────
    def _proxima_rodada(self) -> None:
        r = self.rodada
        self.rodada = None
        self.est.bolas += r.bolas_ganhas
        self.moedas += r.moedas
        # tudo desce uma fileira; bloco na última fileira = game over
        if any(f >= ULTIMA_FILEIRA for b in self.est.blocos for _, f in celulas_de(b)):
            self.game_over = True
            return
        for b in self.est.blocos:
            b.fileira += 1
        for p in self.est.powerups:
            p.fileira += 1
        self.est.powerups = [p for p in self.est.powerups if p.fileira <= ULTIMA_FILEIRA]
        self.est.rodada += 1
        self.gerador.nova_fileira(self.est)

    # ── o que aparece na tela ───────────────────────────────────────────────
    def tela(self) -> Estado:
        """Cópia do estado pra desenhar (o desenhista não pode ver a rodada
        mudando no meio do desenho)."""
        e = copy.copy(self.est)
        e.blocos = [copy.copy(b) for b in self.est.blocos]
        e.powerups = [copy.copy(p) for p in self.est.powerups]
        e.game_over = self.game_over
        r = self.rodada
        if r is None:
            e.em_rodada, e.bolas_voando, e.robo_x = False, [], None
            return e
        e.em_rodada = True
        e.robo_x = r.partida_x  # o robô fica onde a rodada começou
        voando = [(b.x, b.y) for b in r.bolas if b.voando]
        if r.primeira_pousou is not None:  # as que pousaram juntam na 1a
            voando.append((r.primeira_pousou, fis.CHAO_Y))
        e.bolas_voando = voando
        return e
