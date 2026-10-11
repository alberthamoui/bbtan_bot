"""
validar_desenho_local.py — Confere se o jogo local DESENHA igual ao jogo real.

Pra cada captura limpa (logs/captura_analise_N/referencia/tela_tNNN.png):
lê o estado (blocos, power-ups, bola, contador, rodada) com
tools/analisar_capturas.py + sessao.log, redesenha com jogo_local.desenho e
compara as duas imagens:
  - diferença média por pixel no tabuleiro (0-255);
  - o que a VISÃO DO BOT lê em cada uma (blocos, power-ups, bola, contador);
  - lado_tNNN.png: real | desenhado | diferença, pras rodadas pedidas.

Por padrão o bloco é pintado com a cor real medida (isola geometria e texto);
--cor-modelo usa o degradê do desenho.py.

  python tools/validar_desenho_local.py                       # captura_analise_2
  python tools/validar_desenho_local.py logs/captura_analise_5 --lado 10 40
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import config as cfg
import vision as v
import analisar_capturas as ac
from capturar_jogo import MARGEM_CIMA, MARGEM_X
from jogo_local import geometria as geo
from jogo_local.desenho import Desenhista
from jogo_local.estado import Bloco, Estado, PowerUp

BX, BY = cfg.px(MARGEM_X), cfg.px(MARGEM_CIMA)
GW, GH = cfg.GAME_REGION[2], cfg.GAME_REGION[3]


def ler_log(pasta: Path):
    log = (pasta / "sessao.log").read_text(encoding="utf-8", errors="ignore")
    bolas, cont = {}, {}
    for parte in re.split(r"===== TURNO ", log)[1:]:
        t = int(parte.split()[0])
        m = re.search(r"bola x=(\d+) y=(\d+)", parte)
        if m:
            bolas[t] = int(m.group(1))
        m = re.search(r"bolas \(OCR\): (\d+)", parte)
        if m:
            cont[t] = int(m.group(1))
    return bolas, cont


def estado_da_captura(tela: np.ndarray, t: int, bola_x: float, n_bolas: int, cor_real: bool) -> Estado:
    board = tela[BY : BY + GH, BX : BX + GW]
    a = ac.analisar_tabuleiro(board, t)
    blocos = []
    for b in a["blocos"]:
        if b["tipo"].endswith("?"):
            continue
        x0, y0 = int(round(geo.x_coluna(b["coluna"]))), int(round(geo.y_fileira(b["fileira"])))
        # cor dentro do corpo e longe do texto (nos triângulos, a lateral do ângulo reto)
        if b["tipo"] in ("Ttr", "Tbr"):
            cel = board[y0 + 24 : y0 + 36, x0 + 52 : x0 + 57]
        elif b["tipo"] in ("Ttl", "Tbl"):
            cel = board[y0 + 24 : y0 + 36, x0 + 3 : x0 + 8]
        else:
            cel = board[y0 + 4 : y0 + 12, x0 + 4 : x0 + 12]
        cor = tuple(int(c) for c in np.median(cel.reshape(-1, 3), axis=0)) if cor_real else None
        nasceu = t - b["fileira"] + 1
        base = nasceu * (3 if b["tipo"] == "2x2" else 2 if b["marcador"] == "corrente" else 1)
        blocos.append(Bloco(
            coluna=b["coluna"], fileira=b["fileira"], tipo=b["tipo"], hp=max(1, b["hp"]), hp_base=base,
            divisor=b["marcador"] == "divisor", corrente=0 if b["marcador"] == "corrente" else None, cor=cor,
        ))
    pus = [PowerUp(coluna=p["coluna"], fileira=p["fileira"], tipo=p["tipo"]) for p in a["powerups"]]
    return Estado(rodada=t, blocos=blocos, powerups=pus, bola_x=bola_x, bolas=n_bolas)


def leitura_da_visao(img: np.ndarray):
    """O que o bot lê: blocos (fileira, coluna, tipo, HP), power-ups, bola e contador."""
    board = img[BY : BY + GH, BX : BX + GW]
    blocos = sorted((7 - b.row_index, b.col_index, b.block_type.value[:3], b.hp) for b in v.detect_blocks(board))
    pus = sorted((round(p.cy / 64), v._get_col_index(p.cx), p.power_up_type.value) for p in v.detect_powerups(board))
    bola = v.detect_ball(board)
    bx, by, bw, bh = cfg.BALL_COUNT_REGION
    gx, gy = cfg.GAME_REGION[:2]
    cont = v.ler_contador(img[BY + by - gy : BY + by - gy + bh, BX + bx - gx : BX + bx - gx + bw])
    return blocos, pus, (round(bola[0]), round(bola[1])) if bola else None, cont


def blocos_iguais(real, des) -> bool:
    """Mesmas posições/tipos; HP igual, exceto quando o real não leu (0)."""
    forma = lambda bl: sorted((f, c, tipo) for f, c, tipo, _ in bl)
    return forma(real) == forma(des) and all(a[3] == b[3] or a[3] == 0 for a, b in zip(real, des))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", nargs="?", default=str(Path(__file__).resolve().parent.parent / "logs" / "captura_analise_2"))
    ap.add_argument("--cor-modelo", action="store_true", help="pinta os blocos pelo degradê do desenho.py, não pela cor real")
    ap.add_argument("--lado", type=int, nargs="*", default=[9, 43, 64], help="rodadas pra salvar real | desenhado | diferença")
    args = ap.parse_args()
    pasta = Path(args.pasta)
    bolas, cont = ler_log(pasta)
    des = Desenhista()

    total = {"blocos": 0, "powerups": 0, "bola": 0, "contador": 0}
    difs, diverg = [], []
    telas = sorted((pasta / "referencia").glob("tela_t*.png"))
    for p in telas:
        t = int(re.search(r"t(\d+)\.png", p.name).group(1))
        real = cv2.imread(str(p))
        est = estado_da_captura(real, t, bolas.get(t, 226), cont.get(t, t), cor_real=not args.cor_modelo)
        img = des.desenhar(est)
        difs.append(np.abs(img[BY : BY + GH, BX : BX + GW].astype(int) - real[BY : BY + GH, BX : BX + GW].astype(int)).mean())
        lr, ld = leitura_da_visao(real), leitura_da_visao(img)
        ok = [blocos_iguais(lr[0], ld[0]), lr[1] == ld[1], lr[2] == ld[2], lr[3] == ld[3]]
        for nome, o in zip(total, ok):
            total[nome] += o
        if not ok[0]:
            diverg.append((t, sorted(set(lr[0]) - set(ld[0])), sorted(set(ld[0]) - set(lr[0]))))
        if t in args.lado:
            dif = np.clip(np.abs(img.astype(int) - real.astype(int)).max(axis=2) * 3, 0, 255).astype(np.uint8)
            cv2.imwrite(str(pasta / f"lado_t{t:03d}.png"), np.hstack([real, img, cv2.applyColorMap(dif, cv2.COLORMAP_INFERNO)]))

    n = len(telas)
    print(f"{pasta.name}: {n} capturas | diferenca media no tabuleiro {np.mean(difs):.2f}/255 (pior {np.max(difs):.1f})")
    print("visao do bot le igual no real e no desenhado: " + ", ".join(f"{k} {v}/{n}" for k, v in total.items()))
    for t, so_real, so_des in diverg:
        print(f"  t{t}: so no real {so_real} | so no desenhado {so_des}")


if __name__ == "__main__":
    main()
