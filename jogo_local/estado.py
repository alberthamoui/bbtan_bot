"""
estado.py — O que existe na tela do jogo num instante. É o que o desenhista
(desenho.py) transforma em imagem; a lógica das rodadas (etapa 2) é quem vai
alterar isto ao longo da partida.

Coordenadas em células da grade (coluna 0-6, fileira 0-7: 0 = acima da grade
visível, 1-7 = L1-L7 do log do bot) e, pra bolas, em px do tabuleiro.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TIPOS_BLOCO = ("Q", "Ttl", "Ttr", "Tbl", "Tbr", "2x2")  # T = triângulo, letra = canto do ângulo reto
TIPOS_POWERUP = ("+", "LH", "LV", "R", "B")  # LH ◀▶ fileira, LV ▲▼ coluna, R direção aleatória, B moeda


@dataclass
class Bloco:
    coluna: int
    fileira: int  # 2x2: fileira/coluna do canto superior esquerdo
    tipo: str = "Q"
    hp: int = 1
    hp_base: int = 1  # HP de referência da cor (rodada em que nasceu; o 2x2 e a corrente usam o próprio HP inicial)
    divisor: bool = False  # mini bloco no canto (ao quebrar vira 2 blocos com metade do HP)
    corrente: Optional[int] = None  # id do par "Share Block" (os dois dividem o HP)
    cor: Optional[Tuple[int, int, int]] = None  # BGR forçado (validação contra captura real)


@dataclass
class PowerUp:
    coluna: int
    fileira: int
    tipo: str = "+"


@dataclass
class Estado:
    rodada: int = 1
    blocos: List[Bloco] = field(default_factory=list)
    powerups: List[PowerUp] = field(default_factory=list)
    bola_x: float = 226.0  # centro da bola parada (px do tabuleiro); o y é fixo (geometria.BOLA_Y)
    bolas: int = 1  # número no contador "xN"
    em_rodada: bool = False  # bolas voando: a barra de baixo vira Retrieve + acelerar
    bolas_voando: List[Tuple[float, float]] = field(default_factory=list)  # centros (px do tabuleiro)
    robo_x: Optional[float] = None  # onde o robô está (None = junto da bola parada)
    mostrar_robo: bool = True
    game_over: bool = False  # tela de fim: um painel branco (a visão do bot procura >= 25% de branco)
