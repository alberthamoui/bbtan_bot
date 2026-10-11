"""
validar_rodadas_local.py — O jogo local termina a rodada IGUAL ao jogo real?

Juiz: a captura real do turno seguinte (tela_t{N+1}.png). Pra cada rajada
(ângulo e instante de saída medidos nela, tools/medir_rajadas.py), joga a
rodada INTEIRA no jogo local (todas as bolas, até a última pousar) e compara
com o que o jogo real mostrou depois:
  - HP de cada bloco que já existia (na captura seguinte ele está 1 fileira
    abaixo; a fileira nova fica de fora), lido por síntese com a fonte do
    jogo (tools/ler_hp_exato.py) nas duas pontas;
  - quais blocos quebraram;
  - onde a próxima rodada começa (x da bola) e quantas bolas ganhou.
O mesmo pro physics.py do bot (hp_final da simulação, com o que a visão
dele lê). Rodada que passou num R não dá pra comparar (direção sorteada) e
é marcada.

  python tools/validar_rodadas_local.py
"""

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import config as cfg
import analisar_capturas as ac
import medir_rajadas as mr
from ler_hp_exato import ler_hp
from validar_desenho_local import estado_da_captura, ler_log
from validar_fisica_local import TESTE_DESDE, tela_antes
from jogo_local.jogo import Rodada


def tela_depois(rajada: Path):
    t = mr.turno_de(rajada) + 1
    q = [p for p in rajada.parent.glob(f"tela_*t{t:03d}.png") if "rajada" not in p.name]
    return cv2.imread(str(q[0])) if q else None


def blocos_lidos(tela, rodada):
    """{(coluna, fileira): (HP, tipo)} da captura, com o HP lido por síntese."""
    board = tela[160 : 160 + cfg.GAME_REGION[3], 60 : 60 + cfg.GAME_REGION[2]]
    return {(b["coluna"], b["fileira"]): (ler_hp(board, b["coluna"], b["fileira"], b["tipo"], b["marcador"] == "divisor",
                                                 maximo=3 * rodada + 5), b["tipo"])
            for b in ac.analisar_tabuleiro(board, rodada)["blocos"] if not b["tipo"].endswith("?")}


def comparar(antes: dict, previsto: dict, real: dict):
    """antes/previsto/real: {(coluna, fileira): hp} na posição de ANTES da rodada.
    Devolve (blocos com HP exato, blocos comparados, quebras certas, quebras
    reais+previstas, |erro| médio de HP)."""
    exatos = n = 0
    erros = []
    for k, hp0 in antes.items():
        hr, hpv = real.get(k, 0), previsto.get(k, 0)
        if hr == -1:  # OCR do real não leu: não dá pra julgar
            continue
        n += 1
        exatos += hr == hpv
        erros.append(abs(hr - hpv))
    q_real = {k for k in antes if real.get(k, 0) == 0}
    q_prev = {k for k in antes if previsto.get(k, 0) == 0}
    return exatos, n, len(q_real & q_prev), len(q_real | q_prev), float(np.mean(erros)) if erros else 0.0


def main():
    import physics
    import vision as v

    tot = {"treino": np.zeros(5), "teste": np.zeros(5)}
    tot_bot = {"treino": np.zeros(5), "teste": np.zeros(5)}
    bola = {"treino": [], "teste": []}
    contador = {"treino": [], "teste": []}
    print(f"{'rajada':<19}{'grupo':<7}| {'LOCAL: HP exato':>16} {'quebras':>8} {'erro HP':>8} {'x bola':>7} | "
          f"{'BOT: HP exato':>14} {'quebras':>8} {'erro HP':>8}")
    for raj in mr.rajadas():
        depois = tela_depois(raj)
        saida = mr.medir_saida(mr.carregar(raj))
        if depois is None or saida is None:
            continue
        t = mr.turno_de(raj)
        tela = tela_antes(raj)
        bolas_log, cont = ler_log(raj.parent.parent)
        n_bolas = cont.get(t, t)
        est = estado_da_captura(tela, t, saida["partida"][0], n_bolas, cor_real=False, hp_exato=True)
        est.bola_x = saida["partida"][0]
        antes = {(b.coluna, b.fileira): b.hp for b in est.blocos}
        tem_r = any(p.tipo == "R" for p in est.powerups)
        rod = Rodada(est, saida["angulo"], rng=random.Random(0))
        passou_r = False
        ids_r = {id(p) for p in est.powerups if p.tipo == "R"}
        for _ in range(50 * 120):
            rod.passo()
            passou_r |= any(i in b.dentro_de for b in rod.bolas for i in ids_r)
            if rod.acabou:
                break
        previsto = {(b.coluna, b.fileira): b.hp for b in est.blocos}
        lido = blocos_lidos(depois, t + 1)
        # 1 fileira acima no turno anterior; a fileira nova (1) fica de fora,
        # menos o 2x2 que nasceu na fileira 0 e agora está na 1
        real = {(c, f - 1): hp for (c, f), (hp, tipo) in lido.items() if f >= 2 or (f == 1 and tipo == "2x2")}
        r_local = comparar(antes, previsto, real)

        # bot: mesma entrada, o que a visão dele lê
        board = tela[160 : 160 + cfg.GAME_REGION[3], 60 : 60 + cfg.GAME_REGION[2]]
        sim = physics.simulate_trajectory(saida["partida"][0], saida["partida"][1], saida["angulo"],
                                          v.detect_blocks(board), v.detect_powerups(board),
                                          ball_count=n_bolas, record_path=True)
        prev_bot = {(c, 7 - r): hpf for r, c, _, hpf in sim.hp_final}
        r_bot = comparar(antes, {k: prev_bot.get(k, antes[k]) for k in antes}, real)

        cont_ok = cont.get(t + 1) == n_bolas + rod.bolas_ganhas if cont.get(t + 1) is not None else None
        x_real = bolas_log.get(t + 1)
        dx = abs(est.bola_x - x_real) if x_real is not None else float("nan")
        grupo = "teste" if raj.parent.parent.name == "captura_analise_2" and t >= TESTE_DESDE else "treino"
        aviso = "  (passou num R)" if passou_r else ("  (tem R, nao passou)" if tem_r else "")
        nome = f"{raj.parent.parent.name[-1]}/{raj.name}"
        print(f"{nome:<19}{grupo:<7}| {r_local[0]:>9}/{r_local[1]:<6} {r_local[2]:>3}/{r_local[3]:<4} {r_local[4]:>7.1f} "
              f"{dx:>6.0f}px | {r_bot[0]:>7}/{r_bot[1]:<6} {r_bot[2]:>3}/{r_bot[3]:<4} {r_bot[4]:>7.1f}{aviso}")
        if passou_r:
            continue
        tot[grupo] += np.array(r_local)
        tot_bot[grupo] += np.array(r_bot)
        if cont_ok is not None:
            contador[grupo].append(cont_ok)
        if dx == dx:
            bola[grupo].append(dx)

    for g in ("treino", "teste"):
        a, b = tot[g], tot_bot[g]
        if not a[1]:
            continue
        print(f"\n{g.upper()} (sem as rodadas que passaram num R)")
        print(f"  LOCAL: HP exato em {a[0] / a[1]:.0%} dos blocos | quebras certas {a[2]:.0f} de {a[3]:.0f} | "
              f"x da proxima bola a <= 5px em {np.mean(np.array(bola[g]) <= 5):.0%} das rodadas | "
              f"bolas da proxima rodada certas em {np.mean(contador[g]):.0%}")
        print(f"  BOT:   HP exato em {b[0] / b[1]:.0%} dos blocos | quebras certas {b[2]:.0f} de {b[3]:.0f}")


if __name__ == "__main__":
    main()
