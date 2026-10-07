"""
imagem.py — Print da tela com tudo que o config.py marca como posição.

Tira um print da área do jogo (tabuleiro + HUD em volta) e desenha por cima
onde o config.py ACHA que cada coisa está:
  - GAME_REGION (verde)            -> tools/calibrate.py region
  - paredes WALL_* (vermelho)      -> config.py
  - colunas COL_LEFTS/RIGHTS       -> tools/calibrate_columns.py
  - fileiras LEVEL_TOPS/BOTS       -> tools/calibrate_levels.py
  - HUD: contador de bolas, botão retrieve, regiões de fim de rodada,
    clique do retrieve e botão de acelerar -> config.py (regiao_hud/ponto_hud)

Se o desenho não bater com o jogo (grade fora dos blocos, retângulo do retrieve
fora do botão roxo...), a legenda à direita diz qual variável ajustar e com qual
ferramenta. Depois de mudar a janela do jogo de lugar, normalmente basta corrigir
o GAME_REGION: o resto é relativo a ele e acompanha.

Uso:
  python imagem.py                -> print + janela (qualquer tecla fecha)
  python imagem.py --delay 3      -> espera 3s antes do print (pra focar o jogo)
  python imagem.py --sem-janela   -> só salva em prints/
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np

import config as cfg

SAIDA = Path(__file__).resolve().parent / "prints"

ZOOM = 1.5  # o print é ampliado antes de desenhar, pra linha fina ficar legível
MARGEM = 30  # px de tela em volta de tudo que é desenhado
LARGURA_LEGENDA = 520

# Cores (BGR)
COR_REGIAO = (0, 255, 0)  # verde
COR_PAREDE = (0, 0, 255)  # vermelho
COR_COLUNA = (255, 255, 0)  # ciano
COR_FILEIRA = (0, 255, 255)  # amarelo
COR_CONTADOR = (255, 0, 255)  # magenta
COR_RETRIEVE = (0, 165, 255)  # laranja
COR_FIM_RODADA = (203, 192, 255)  # rosa
COR_CLIQUE = (255, 255, 255)  # branco


def _texto(img, texto, org, cor, escala=0.45, grossura=1):
    """Texto com contorno preto. Só ASCII: o putText não renderiza acento."""
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grossura + 2)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura)


def _area_do_print() -> dict:
    """Retângulo de tela (absoluto) que cobre o GAME_REGION + todo o HUD."""
    regioes = [cfg.GAME_REGION, cfg.BALL_COUNT_REGION, cfg.RETRIEVE_BUTTON_REGION]
    regioes += list(cfg.ROUND_END_REGIONS)
    pontos = [cfg.RETRIEVE_CLICK, (cfg.SPEED_UP_X, cfg.SPEED_UP_Y)]

    xs = [r[0] for r in regioes] + [r[0] + r[2] for r in regioes] + [p[0] for p in pontos]
    ys = [r[1] for r in regioes] + [r[1] + r[3] for r in regioes] + [p[1] for p in pontos]
    return {
        "left": int(min(xs)) - MARGEM,
        "top": int(min(ys)) - MARGEM,
        "width": int(max(xs) - min(xs)) + 2 * MARGEM,
        "height": int(max(ys) - min(ys)) + 2 * MARGEM,
    }


def _capturar(area: dict) -> np.ndarray:
    """Print da área pedida. Pedaços fora de qualquer monitor ficam pretos."""
    with mss.mss() as sct:
        virtual = sct.monitors[0]
        x1 = max(area["left"], virtual["left"])
        y1 = max(area["top"], virtual["top"])
        x2 = min(area["left"] + area["width"], virtual["left"] + virtual["width"])
        y2 = min(area["top"] + area["height"], virtual["top"] + virtual["height"])

        frame = np.zeros((area["height"], area["width"], 3), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            shot = sct.grab({"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1})
            ox, oy = x1 - area["left"], y1 - area["top"]
            frame[oy : oy + (y2 - y1), ox : ox + (x2 - x1)] = np.array(shot)[:, :, :3]
    return frame


def desenhar(frame: np.ndarray, area: dict) -> np.ndarray:
    img = cv2.resize(frame, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

    def p(x, y):
        """Coordenada absoluta de tela -> pixel na imagem ampliada."""
        return (int(round((x - area["left"]) * ZOOM)), int(round((y - area["top"]) * ZOOM)))

    def retangulo(r, cor, grossura=1):
        x, y, w, h = r
        cv2.rectangle(img, p(x, y), p(x + w, y + h), cor, grossura)

    gx, gy, gw, gh = cfg.GAME_REGION

    # GAME_REGION: 1px pra fora, pra não cobrir as paredes (WALL_LEFT = 0)
    cv2.rectangle(img, p(gx - 1, gy - 1), p(gx + gw, gy + gh), COR_REGIAO, 2)
    _texto(img, "GAME_REGION", (p(gx, gy)[0], p(gx, gy)[1] - 6), COR_REGIAO)

    # Paredes da física
    cv2.rectangle(
        img,
        p(gx + cfg.WALL_LEFT, gy + cfg.WALL_TOP),
        p(gx + cfg.WALL_RIGHT, gy + cfg.WALL_BOTTOM),
        COR_PAREDE,
        1,
    )

    # Grade: colunas (vertical) e fileiras (horizontal), só na altura/largura da grade
    topo, base = gy + cfg.LEVEL_TOPS[-1], gy + cfg.LEVEL_BOTS[0]
    esq, dir_ = gx + cfg.COL_LEFTS[0], gx + cfg.COL_RIGHTS[-1]
    for i, (l, r) in enumerate(zip(cfg.COL_LEFTS, cfg.COL_RIGHTS)):
        for x in (gx + l, gx + r):
            cv2.line(img, p(x, topo), p(x, base), COR_COLUNA, 1)
        cx = p(gx + (l + r) / 2, topo)[0]
        _texto(img, f"C{i}", (cx - 8, p(0, topo)[1] - 6), COR_COLUNA, 0.4)
    for i, (t, b) in enumerate(zip(cfg.LEVEL_TOPS, cfg.LEVEL_BOTS)):
        for y in (gy + t, gy + b):
            cv2.line(img, p(esq, y), p(dir_, y), COR_FILEIRA, 1)
        cy = p(esq, gy + (t + b) / 2)[1]
        _texto(img, f"L{i}", (p(gx, 0)[0] - 26, cy + 5), COR_FILEIRA, 0.4)

    # HUD
    retangulo(cfg.BALL_COUNT_REGION, COR_CONTADOR, 2)
    bx, by, _, bh = cfg.BALL_COUNT_REGION
    _texto(img, "BALL_COUNT_REGION", (p(bx, by)[0] + 4, p(bx, by + bh)[1] + 14), COR_CONTADOR, 0.4)

    retangulo(cfg.RETRIEVE_BUTTON_REGION, COR_RETRIEVE, 2)
    rx, ry, _, rh = cfg.RETRIEVE_BUTTON_REGION
    _texto(img, "RETRIEVE_BUTTON_REGION", (p(rx, ry)[0], p(rx, ry + rh)[1] + 14), COR_RETRIEVE, 0.4)

    for i, r in enumerate(cfg.ROUND_END_REGIONS):
        retangulo(r, COR_FIM_RODADA, 1)
        _texto(img, f"E{i}", (p(r[0], r[1])[0], p(r[0], r[1])[1] - 4), COR_FIM_RODADA, 0.35)

    for nome, (x, y) in (
        ("RETRIEVE_CLICK", cfg.RETRIEVE_CLICK),
        ("SPEED_UP", (cfg.SPEED_UP_X, cfg.SPEED_UP_Y)),
    ):
        c = p(x, y)
        cv2.drawMarker(img, c, COR_CLIQUE, cv2.MARKER_CROSS, 18, 2)
        cv2.circle(img, c, 9, COR_CLIQUE, 1)
        _texto(img, nome, (c[0] + 14, c[1] + 5), COR_CLIQUE, 0.4)

    return img


def _legenda(altura: int) -> np.ndarray:
    """Painel à direita: cor, variável, valor atual e como calibrar."""
    painel = np.full((altura, LARGURA_LEGENDA, 3), 25, dtype=np.uint8)
    gx, gy, gw, gh = cfg.GAME_REGION
    escala = getattr(cfg, "ESCALA", gw / 452)

    itens = [
        (COR_REGIAO, "GAME_REGION", f"({gx}, {gy}, {gw}, {gh})", "tools/calibrate.py region"),
        (
            COR_PAREDE,
            "WALL_LEFT/RIGHT/TOP/BOTTOM",
            f"{cfg.WALL_LEFT}, {cfg.WALL_RIGHT}, {cfg.WALL_TOP}, {cfg.WALL_BOTTOM}",
            "config.py (relativo ao GAME_REGION)",
        ),
        (COR_COLUNA, "COL_LEFTS / COL_RIGHTS (C0-C6)", str(cfg.COL_LEFTS), "tools/calibrate_columns.py"),
        (COR_FILEIRA, "LEVEL_TOPS / LEVEL_BOTS (L0-L6)", str(cfg.LEVEL_TOPS), "tools/calibrate_levels.py"),
        (COR_CONTADOR, "BALL_COUNT_REGION", str(cfg.BALL_COUNT_REGION), "config.py (regiao_hud)"),
        (COR_RETRIEVE, "RETRIEVE_BUTTON_REGION", str(cfg.RETRIEVE_BUTTON_REGION), "config.py (regiao_hud)"),
        (COR_FIM_RODADA, "ROUND_END_REGIONS (E0-E3)", "4 regioes", "config.py (regiao_hud)"),
        (
            COR_CLIQUE,
            "RETRIEVE_CLICK / SPEED_UP_X,Y",
            f"{cfg.RETRIEVE_CLICK} / ({cfg.SPEED_UP_X}, {cfg.SPEED_UP_Y})",
            "config.py (ponto_hud) + tools/get_position.py",
        ),
    ]

    y = 28
    _texto(painel, f"ESCALA {escala:.3f}  (largura do jogo / 452)", (12, y), (230, 230, 230), 0.5)
    y += 34
    for cor, nome, valor, ferramenta in itens:
        cv2.rectangle(painel, (12, y - 12), (26, y + 2), cor, -1)
        _texto(painel, nome, (36, y), cor, 0.45)
        _texto(painel, valor, (36, y + 18), (210, 210, 210), 0.4)
        _texto(painel, f"-> {ferramenta}", (36, y + 35), (150, 150, 150), 0.4)
        y += 58

    y += 6
    for linha in (
        "Conferir:",
        "- verde colado na borda do tabuleiro",
        "- ciano/amarelo nas bordas dos blocos",
        "- laranja em cima do botao roxo Retrieve",
        "  (so aparece com as bolas voando)",
        "- magenta em volta do contador 'x14'",
        "- cruz branca no Retrieve e no acelerar",
        "Mudou a janela de lugar? Corrija so o",
        "GAME_REGION: o resto e relativo a ele.",
    ):
        _texto(painel, linha, (12, y), (230, 230, 230), 0.42)
        y += 20
    return painel


def main():
    parser = argparse.ArgumentParser(
        description="Print da tela com as posicoes do config.py desenhadas"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Segundos de espera antes do print"
    )
    parser.add_argument(
        "--sem-janela", action="store_true", help="So salva em prints/, sem abrir janela"
    )
    args = parser.parse_args()

    if args.delay > 0:
        print(f"Print em {args.delay:.0f}s...")
        time.sleep(args.delay)

    area = _area_do_print()
    img = desenhar(_capturar(area), area)
    painel = _legenda(img.shape[0])
    if painel.shape[0] > img.shape[0]:  # legenda maior que o print: completa com preto
        img = cv2.copyMakeBorder(img, 0, painel.shape[0] - img.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    img = np.hstack([img, painel[: img.shape[0]]])

    SAIDA.mkdir(exist_ok=True)
    arquivo = SAIDA / f"calibracao_{datetime.now():%Y%m%d_%H%M%S}.png"
    if not cv2.imwrite(str(arquivo), img):
        print(f"[!] Falha ao salvar {arquivo}")
        return
    print(f"GAME_REGION = {cfg.GAME_REGION}")
    print(f"Salvo em prints/{arquivo.name}")

    if not args.sem_janela:
        cv2.imshow("Calibracao: posicoes do config.py", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
