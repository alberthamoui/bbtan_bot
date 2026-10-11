"""
geometria.py — Onde cada coisa fica no jogo real, MEDIDO nas capturas limpas
(logs/captura_analise_2, com ESCALA = 1). Coordenadas do TABULEIRO (0,0 = canto
do GAME_REGION), em px; o resto do bot usa as mesmas.

O passo da grade NÃO é o BLOCK_SIZE = 64 do config. Ajustado nas bordas
FRACIONÁRIAS dos quadrados roxos (cobertura do pixel da borda, 89 blocos das
rodadas 31-71): x = 5,60 + 63,42·c, y = 70,06 + 63,48·(f - 1), lado 59,77
(com a sombra preta de 1 px embaixo/à direita por fora). A fileira 0 fica ACIMA
da grade visível: é onde o 2x2 recém-nascido põe a metade de cima.
"""

import config as cfg

COLUNAS = 7
FILEIRAS = 8  # 0 (acima da grade) + L1..L7

# antigo (inteiros, só pelo 1o pixel colorido): passo 63,5 x 63,33, X0 5, Y1 70, lado 60,3
PASSO_X = 63.42 * cfg.ESCALA
PASSO_Y = 63.48 * cfg.ESCALA
X0 = 5.60 * cfg.ESCALA  # canto esquerdo da coluna 0
Y1 = 70.06 * cfg.ESCALA  # topo da fileira 1 (L1)
LADO = 59.77 * cfg.ESCALA  # lado do quadrado
LADO_2X2 = LADO + PASSO_X  # 2x2 cobre duas células e o vão entre elas

# Bola parada (centro) — o y não muda; o x é onde a 1a bola caiu
BOLA_Y = 566.0 * cfg.ESCALA
RAIO_BOLA = 9.0 * cfg.ESCALA

# Texto do HP: altura dos dígitos (fonte igual em 1x1 e 2x2), centrado no
# bloco 0,5 px acima do meio. Nos triângulos, encostado no lado do ângulo
# reto: topo do texto a 3 px da borda (cantos de cima) ou a 41 px (cantos de
# baixo), centro x a 21 px (cantos da esquerda) ou 38,5 px (da direita).
HP_ALTURA = 15
HP_DY_CENTRO = -0.5
TRI_TEXTO_Y = {"t": 3.0, "b": 41.0}
TRI_TEXTO_CX = {"l": 21.0, "r": 38.5}


# Bordas medidas POR coluna/fileira (o jogo arredonda as posições do jeito
# dele: a largura varia de 59,55 a 59,95 px conforme a coluna). Fora da
# tabela (fileira 0, 6, 7), a reta ajustada acima.
BORDA_ESQ = {0: 5.58, 1: 69.0, 2: 132.45, 3: 196.0, 4: 259.0, 5: 323.0, 6: 386.0}
BORDA_DIR = {0: 65.39, 1: 128.78, 2: 192.0, 3: 255.69, 4: 318.95, 5: 382.58, 6: 445.88}
BORDA_TOPO = {1: 70.0, 2: 133.64, 3: 197.0, 4: 260.41, 5: 324.0}
BORDA_BAIXO = {1: 129.89, 2: 193.4, 3: 256.76, 4: 319.96, 5: 383.58}
# Desligado: com a tabela a visão do bot lê os blocos desenhados iguais aos
# reais em 62 de 71 capturas, contra 65 com a reta (as bordas da tabela foram
# medidas sem descontar a sombra preta e ficaram enviesadas).
USAR_TABELA = False


def x_coluna(c: float) -> float:
    if USAR_TABELA and c in BORDA_ESQ:
        return BORDA_ESQ[c] * cfg.ESCALA
    return X0 + PASSO_X * c


def y_fileira(f: float) -> float:
    if USAR_TABELA and f in BORDA_TOPO:
        return BORDA_TOPO[f] * cfg.ESCALA
    return Y1 + PASSO_Y * (f - 1)


def largura_coluna(c: int) -> float:
    if USAR_TABELA and c in BORDA_ESQ:
        return (BORDA_DIR[c] - BORDA_ESQ[c]) * cfg.ESCALA
    return LADO


def altura_fileira(f: int) -> float:
    if USAR_TABELA and f in BORDA_TOPO:
        return (BORDA_BAIXO[f] - BORDA_TOPO[f]) * cfg.ESCALA
    return LADO


def centro_celula(c: float, f: float):
    return x_coluna(c) + LADO / 2, y_fileira(f) + LADO / 2


# Bloco divisor (mini bloco no canto), medido em 50 ocorrências, relativo ao
# canto da célula (x_coluna, y_fileira): o corpo fica deslocado; o mini bloco
# (22x22 com o contorno) sai um pouco acima/à direita; o contorno é um L preto
# de 2 px, à esquerda e embaixo do mini bloco, por cima do corpo.
DIV_CORPO_DESLOC = (-1.03 * cfg.ESCALA, 0.88 * cfg.ESCALA)
DIV_MINI = (38.7 * cfg.ESCALA, -1.06 * cfg.ESCALA, 60.7 * cfg.ESCALA, 20.94 * cfg.ESCALA)  # x0, y0, x1, y1
DIV_CONTORNO = 2.0 * cfg.ESCALA

