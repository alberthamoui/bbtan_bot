"""
validar_fisica_local.py — O jogo local MEXE a bola igual ao jogo real?

Juiz: as rajadas gravadas do jogo real (60 fps). Pra cada uma:
  - as bolas reais de cada quadro saem de tools/medir_rajadas.py (sem física);
  - o ângulo e o instante de saída são os REAIS (medidos nas bolas do 1o
    trecho), então o viés da mira não entra na conta;
  - o tabuleiro é o da captura antes do tiro (tela_tNNN.png), com o HP lido
    por síntese (tools/ler_hp_exato.py), sem os erros do OCR do bot.

Três vias, mesma entrada:
  REAL  x  LOCAL (jogo_local/fisica.py + jogo.py)
  REAL  x  BOT   (physics.py, com o que a visão do bot lê no tabuleiro)
Métricas, até o clique no acelerar (depois dele o emulador pula quadros e o
relógio por quadro fica impreciso — aí o juiz é validar_rodadas_local.py):
  - 1a bola: até quantos px de caminho ela tem uma bola real a <= 5 px;
  - todas as bolas: por quanto tempo >= 75% delas casam com bolas reais a
    <= 5 px, quadro a quadro, sem precisar saber qual é qual;
  - enquanto fiel: a que distância cada bola fica da real.
Se o LOCAL não ficar claramente mais fiel que o BOT, ele não serve de juiz
pro bot.

Treino x teste: as constantes da física local foram medidas/ajustadas só nas
rajadas de TREINO (turnos 1-36); as de TESTE (41-71) ficam de fora.

  python tools/validar_fisica_local.py
  python tools/validar_fisica_local.py --raio 9 --encolher 3   # testa outro contato
  python tools/validar_fisica_local.py --imagem t031   # desenha real x local x bot
"""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import config as cfg
import medir_rajadas as mr
from validar_desenho_local import estado_da_captura, ler_log
from jogo_local import fisica as fis
from jogo_local.jogo import Rodada

LIMITE_PX = 5.0
TESTE_DESDE = 41  # turnos >= isto (captura_analise_2) são teste


def tela_antes(rajada: Path) -> np.ndarray:
    t = mr.turno_de(rajada)
    q = [p for p in rajada.parent.glob(f"tela_*t{t:03d}.png") if "rajada" not in p.name][0]
    return cv2.imread(str(q))


def previsao_local(tela, rajada, partida, ang, bolas, ticks):
    """Posições por tick: [[(x, y, bola) de cada bola voando]], a trilha da
    1a bola e o tick em que ela passou num R (dali em diante a direção é
    sorteada: não dá pra comparar)."""
    t = mr.turno_de(rajada)
    est = estado_da_captura(tela, t, partida[0], bolas, cor_real=False, hp_exato=True)
    est.bola_x = partida[0]
    rod = Rodada(est, ang, rng=random.Random(0))
    rs = [id(p) for p in est.powerups if p.tipo == "R"]
    quadros, primeira, aleatorio_em = [[]], [(partida[0], fis.CHAO_Y)], None
    for k in range(ticks):
        rod.passo()
        quadros.append([(b.x, b.y, i) for i, b in enumerate(rod.bolas) if b.voando])
        b = rod.bolas[0]
        primeira.append((b.x, b.y))
        if aleatorio_em is None and any(r in b.dentro_de for r in rs):
            aleatorio_em = k + 1
    return quadros, primeira, aleatorio_em


def previsao_bot(tela, partida, ang, bolas, ticks):
    """Mesmo formato, com o physics.py do bot e o que a visão dele lê."""
    import physics
    import vision as v

    board = tela[160 : 160 + cfg.GAME_REGION[3], 60 : 60 + cfg.GAME_REGION[2]]
    r = physics.simulate_trajectory(partida[0], partida[1], ang, v.detect_blocks(board), v.detect_powerups(board),
                                    ball_count=bolas, record_path=True)
    caminhos = []
    for cam in r.ball_paths or [r.path]:
        c = np.array([partida] + list(cam), float)
        caminhos.append((c, np.concatenate([[0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])))

    def em(i, dist):
        c, s = caminhos[min(i, len(caminhos) - 1)]
        if dist > s[-1]:
            return None
        j = max(1, min(int(np.searchsorted(s, dist)), len(s) - 1))
        f = (dist - s[j - 1]) / max(s[j] - s[j - 1], 1e-9)
        return tuple(c[j - 1] + f * (c[j] - c[j - 1]))

    quadros, primeira = [], []
    for k in range(ticks + 1):
        q = []
        for i in range(bolas):
            k_i = k - i * fis.INTERVALO_TICKS
            if k_i <= 0:
                continue
            p = em(i, k_i * fis.PASSO)
            if p is not None and p[1] < fis.CHAO_Y - 0.5:
                q.append((p[0], p[1], i))
        quadros.append(q)
        p0 = em(0, k * fis.PASSO)
        primeira.append(p0 if p0 is not None else (float("nan"), float("nan")))
    return quadros, primeira


def comparar_primeira(rajada, partida, t0, primeira, ate_s, corte=None):
    """
    1a bola prevista contra as bolas reais do quadro: até quantos px de
    caminho ela tem uma bola real a <= 5 px em todo quadro (2 quadros
    seguidos sem = divergiu). Escondida atrás do robô não conta.
    """
    d = mr.carregar(rajada)
    sx = partida[0]
    ruins, ultimo_k = 0, 0
    for t, bs in zip(d["tempos"], d["bolas"]):
        k = int(round((t - t0) / fis.TICK_S))
        if k < 1 or t > ate_s or (corte is not None and k >= corte):
            continue
        perto = False
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < len(primeira) and primeira[kk][0] == primeira[kk][0]:
                x, y = primeira[kk]
                if _escondida(x, y, sx) or any(math.hypot(a - x, b - y) <= LIMITE_PX for a, b in bs):
                    perto = True
        if perto:
            ruins, ultimo_k = 0, k
        else:
            ruins += 1
            if ruins == 2:
                return ultimo_k * fis.PASSO
    return float("inf")  # fiel até o fim


def _escondida(x, y, sx):
    """Atrás do robô (ele fica do lado direito do ponto de partida) ou no chão."""
    return y > 555 or (sx - 12 <= x <= sx + 100 and y > 470)


def _casar(prev, det, dists=None) -> int:
    pares, usados = 0, set()
    for x, y, _ in prev:
        melhor = None
        for j, (a, b) in enumerate(det):
            if j in usados:
                continue
            e = math.hypot(a - x, b - y)
            if e <= LIMITE_PX and (melhor is None or e < melhor[0]):
                melhor = (e, j)
        if melhor:
            usados.add(melhor[1])
            pares += 1
            if dists is not None:
                dists.append(melhor[0])
    return pares


def comparar_quadros(rajada, partida, t0, quadros, ate_s, corte=None):
    """
    Todas as bolas, quadro a quadro, sem precisar saber qual bola é qual: em
    cada quadro gravado (até o acelerar), quantas bolas previstas têm uma bola
    real a <= 5 px (e vice-versa). O tick de cada quadro sai do relógio da 1a
    bola, com folga de ±1 (a captura a 60 fps de um jogo a 50 Hz atrasa até
    1 tick). Devolve (segundos fiéis desde o disparo, fração de quadros bons,
    segundos comparados).
    """
    d = mr.carregar(rajada)
    sx, sy = partida
    bons, fiel_ate, ruins_seguidos, n = 0, None, 0, 0
    distancias = []
    for t, bs in zip(d["tempos"], d["bolas"]):
        k = int(round((t - t0) / fis.TICK_S))
        if k < 1 or t > ate_s:
            continue
        if corte is not None and k >= corte:
            break
        det = [(x, y) for x, y in bs if not _escondida(x, y, sx) and math.hypot(x - sx, y - sy) > 3]
        melhor, dist_melhor = -1.0, []
        for kk in (k - 1, k, k + 1):
            if not 0 <= kk < len(quadros):
                continue
            prev = [q for q in quadros[kk] if not _escondida(q[0], q[1], sx)]
            tot = max(len(prev), len(det))
            ds = []
            f = 1.0 if tot == 0 else _casar(prev, det, ds) / tot
            if f > melhor or (f == melhor and np.mean(ds or [0]) < np.mean(dist_melhor or [0])):
                melhor, dist_melhor = f, ds
        if fiel_ate is None:
            distancias += dist_melhor
        n += 1
        if melhor >= 0.75:
            bons += 1
            ruins_seguidos = 0
        else:
            ruins_seguidos += 1
            if ruins_seguidos == 2 and fiel_ate is None:
                fiel_ate = (k - 1) * fis.TICK_S
    ultimo = min(ate_s - t0, (corte or 10**9) * fis.TICK_S)
    return (fiel_ate if fiel_ate is not None else ultimo), bons / max(n, 1), ultimo, distancias


def desenhar(tela, rajada, t0, ate_s, local, bot, saida: Path):
    """Todas as bolas reais detectadas até o acelerar (pontos, do azul ao
    amarelo com o tempo) e o caminho da 1a bola prevista (local e bot)."""
    board = tela[160 : 160 + cfg.GAME_REGION[3], 60 : 60 + cfg.GAME_REGION[2]].copy()
    d = mr.carregar(rajada)
    for t, bs in zip(d["tempos"], d["bolas"]):
        if t0 <= t <= ate_s:
            f = (t - t0) / max(ate_s - t0, 1e-6)
            for x, y in bs:
                cv2.circle(board, (int(round(x)), int(round(y))), 1, (int(255 * (1 - f)), int(200 * f), int(255 * f)), -1)
    for pts, cor in ((bot, (0, 0, 255)), (local, (0, 255, 0))):
        pts = [p for p in pts if p[0] == p[0]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(board, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), cor, 1, cv2.LINE_AA)
    cv2.putText(board, "real (pontos) | local (verde) | bot (vermelho)", (5, 572), 0, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(saida), board)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raio", type=float, help="RAIO_BOLA a testar (px)")
    ap.add_argument("--encolher", type=float, help="ENCOLHER a testar (px)")
    ap.add_argument("--encolher-y", type=float, help="ENCOLHER_Y a testar (px)")
    ap.add_argument("--imagem", nargs="*", default=[], help="turnos pra desenhar (ex: t031)")
    args = ap.parse_args()
    if args.raio is not None:
        fis.RAIO_BOLA = args.raio * cfg.ESCALA
    if args.encolher is not None:
        fis.ENCOLHER = args.encolher * cfg.ESCALA
    if args.encolher_y is not None:
        fis.ENCOLHER_Y = args.encolher_y * cfg.ESCALA

    linhas = {"treino": [], "teste": []}
    dist = {"treino": ([], []), "teste": ([], [])}
    print(f"{'':<19}{'':7}| {'1a BOLA: fiel ate (px de caminho)':^40} | {'TODAS AS BOLAS: segundos fieis':^36}")
    print(f"{'rajada':<19}{'grupo':<7}| {'comparado':>9} {'LOCAL':>14} {'BOT':>14} | {'comparado':>9} {'LOCAL':>12} {'BOT':>12}")
    for raj in mr.rajadas():
        saida = mr.medir_saida(mr.carregar(raj))
        nome = f"{raj.parent.parent.name[-1]}/{raj.name}"
        if saida is None:
            print(f"{nome:<19}sem bolas suficientes pra medir a saida")
            continue
        partida, ang, t0 = saida["partida"], saida["angulo"], saida["t0"]
        tela = tela_antes(raj)
        _, cont = ler_log(raj.parent.parent)
        t = mr.turno_de(raj)
        bolas = cont.get(t, t)
        ate_s = mr.carregar(raj)["eventos"]["acelerar"]
        ticks = int((ate_s - t0) / fis.TICK_S) + 3
        lq, lp, corte = previsao_local(tela, raj, partida, ang, bolas, ticks)
        bq, bp = previsao_bot(tela, partida, ang, bolas, ticks)
        total1 = (min(ticks, corte or ticks) - 1) * fis.PASSO
        lf = min(total1, comparar_primeira(raj, partida, t0, lp, ate_s, corte))
        bf = min(total1, comparar_primeira(raj, partida, t0, bp, ate_s, corte))
        ls, lpct, tot_s, ldist = comparar_quadros(raj, partida, t0, lq, ate_s, corte)
        bs, bpct, _, bdist = comparar_quadros(raj, partida, t0, bq, ate_s, corte)
        grupo = "teste" if raj.parent.parent.name == "captura_analise_2" and t >= TESTE_DESDE else "treino"
        dist[grupo][0].extend(ldist)
        dist[grupo][1].extend(bdist)
        linhas[grupo].append((lf, bf, total1, ls, bs, tot_s, lpct, bpct))
        print(f"{nome:<19}{grupo:<7}| {total1:>7.0f}px {lf:>12.0f}px {bf:>12.0f}px | {tot_s:>8.2f}s {ls:>11.2f}s {bs:>11.2f}s"
              + (f"  (R no tick {corte})" if corte else ""))
        if args.imagem and raj.name.endswith(tuple(args.imagem)):
            desenhar(tela, raj, t0, ate_s, lp, bp, raj.parent.parent / f"fisica_{raj.name}.png")

    for g, ls in linhas.items():
        if not ls:
            continue
        a = np.array(ls)
        print(f"\n{g.upper()} ({len(ls)} rajadas)")
        for nome, i1, i2, ip in (("LOCAL", 0, 3, 6), ("BOT", 1, 4, 7)):
            print(f"  {nome:<5}: 1a bola fiel em {a[:, i1].sum() / a[:, 2].sum():5.1%} do caminho "
                  f"(inteira em {np.mean(a[:, i1] >= a[:, 2] - 1):.0%} das rajadas) | todas as bolas fieis em "
                  f"{a[:, i2].sum() / a[:, 5].sum():5.1%} do tempo, {np.mean(a[:, ip]):.0%} dos quadros")
        for nome, ds in (("LOCAL", dist[g][0]), ("BOT", dist[g][1])):
            print(f"  {nome:<5}: enquanto fiel, cada bola fica a {np.median(ds):.2f}px da real (p90 {np.percentile(ds, 90):.2f}px)")


if __name__ == "__main__":
    main()
