"""
extrair_assets_jogo.py — Recorta, das capturas limpas do jogo real
(logs/captura_analise_N/referencia/), tudo o que o jogo local precisa pra
desenhar a tela igual: fundo, barras de baixo, robô, bola, power-ups, ícones
dos blocos e os DÍGITOS de verdade (HP, contador "xN" e "LINE N").

Saída em jogo_local/assets/ (versionado): PNGs + assets.json com as medidas.
Rode de novo quando houver capturas melhores:

  python tools/extrair_assets_jogo.py                          # usa logs/captura_analise_2
  python tools/extrair_assets_jogo.py logs/captura_analise_5

Coordenadas: tudo relativo à JANELA capturada (MARGEM_X à esquerda e
MARGEM_CIMA acima do canto do GAME_REGION, ver tools/capturar_jogo.py).

Os elementos são guardados como cor + alfa (RGBA) tirados por diferença com
o fundo: num fundo de cor única B, pixel = a*S + (1-a)*B, então com a
estimado pela distância até B dá pra recuperar S e recompor exatamente sobre
o mesmo fundo. Texto preto sobre bloco (HP, corrente, divisor) vira só alfa
(1 - pixel/cor do bloco).
"""

import collections
import json
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

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "jogo_local" / "assets"
BX, BY = cfg.px(MARGEM_X), cfg.px(MARGEM_CIMA)  # canto do tabuleiro dentro da janela
PASSO = cfg.BLOCK_SIZE
ICONE_LADO = 26  # quadrado onde ficam a corrente / o mini bloco do divisor
ICONE_FORA = 3  # quanto esse quadrado passa pra fora do canto do bloco


def _carregar(pasta: Path):
    ref = pasta / "referencia"
    reg = json.load(open(ref / "regioes.json", encoding="utf-8"))
    telas = {}
    for p in sorted(ref.glob("tela_t*.png")):
        t = int(re.search(r"t(\d+)\.png", p.name).group(1))
        telas[t] = cv2.imread(str(p))
    rajadas = {}
    for p in sorted(ref.glob("tela_rajada_t*.png")):
        t = int(re.search(r"t(\d+)\.png", p.name).group(1))
        rajadas[t] = cv2.imread(str(p))
    return telas, rajadas, reg


def _rgba(img: np.ndarray, fundo: np.ndarray, escala: float = 60.0) -> np.ndarray:
    """Elemento claro/colorido sobre o fundo de cor única -> BGRA com alfa pela distância ao fundo."""
    dif = np.abs(img.astype(np.float32) - fundo.astype(np.float32)).max(axis=2)
    a = np.clip(dif / escala, 0, 1)
    cor = fundo.astype(np.float32) + (img.astype(np.float32) - fundo.astype(np.float32)) / np.maximum(a, 1e-3)[..., None]
    out = np.zeros(img.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(cor, 0, 255).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def _alfa_preto(img: np.ndarray, cor_fundo) -> np.ndarray:
    """Tinta preta sobre cor_fundo: alfa = 1 - pixel/cor, no canal mais forte da cor."""
    c = int(np.argmax(cor_fundo))
    return np.clip(1.0 - img[..., c].astype(np.float32) / max(1.0, float(cor_fundo[c])), 0, 1)


def _alfa_branco(img: np.ndarray, cor_fundo) -> np.ndarray:
    """Texto branco sobre cor_fundo: alfa = (pixel - fundo) / (255 - fundo), no canal mais escuro do fundo."""
    c = int(np.argmin(cor_fundo))
    f = float(cor_fundo[c])
    return np.clip((img[..., c].astype(np.float32) - f) / max(1.0, 255.0 - f), 0, 1)


def _media_alinhada(recortes: list) -> np.ndarray:
    """Média de recortes de alfa do mesmo glifo, alinhados pelo canto superior esquerdo (tamanho da moda)."""
    forma = collections.Counter(r.shape for r in recortes).most_common(1)[0][0]
    rs = [r for r in recortes if r.shape == forma]
    return np.mean(rs, axis=0)


# ── fundo e barras ──────────────────────────────────────────────────────────
def extrair_fundo(telas: dict, rajadas: dict, rodada_minima: int):
    """
    Mediana pixel a pixel (blocos, bolas e robô mudam de lugar e somem).
    Só rodadas >= rodada_minima: a barra de itens muda de "Line 30" pra
    "Tap to unlock" na rodada 30. As fileiras 0 e 1 (quase sempre ocupadas)
    vêm de 3 fileiras abaixo (os pontinhos da grade repetem a cada 64 px) e o
    número do "LINE" é apagado (o jogo local desenha por cima).
    """
    tardias = [im for t, im in telas.items() if t >= rodada_minima]
    fundo = np.median(np.stack(tardias), axis=0).astype(np.uint8)
    cor = fundo[BY + 300, BX + 100].copy()
    y0, y1 = BY + cfg.px(2), BY + cfg.px(2) + 2 * PASSO
    fundo[y0:y1, BX + 1 : BX + cfg.GAME_REGION[2] - 1] = fundo[y0 + 3 * PASSO : y1 + 3 * PASSO, BX + 1 : BX + cfg.GAME_REGION[2] - 1]
    # Sobras da mediana dentro do tabuleiro (algo que ficou no mesmo lugar em
    # mais da metade das rodadas, ex: power-up na coluna 6, fileira 2): tudo
    # que não é cor de fundo e é maior que os pontinhos da grade vira fundo.
    # A moldura (borda do tabuleiro) fica de fora.
    gw, gh = cfg.GAME_REGION[2], cfg.GAME_REGION[3]
    m = cfg.px(3)
    interior = fundo[BY + m : BY + gh - m, BX + m : BX + gw - m]
    dif = (np.abs(interior.astype(int) - cor.astype(int)).max(axis=2) > 12).astype(np.uint8)
    n, rot, st, _ = cv2.connectedComponentsWithStats(dif)
    for i in range(1, n):
        if st[i][4] > 10:  # pontinho da grade: ~4 px
            interior[rot == i] = cor
    # número do LINE (o rótulo "LINE" fica acima, em ~69-57 px acima do tabuleiro)
    fundo[BY - cfg.px(55) : BY - cfg.px(4), BX + cfg.px(150) : BX + cfg.px(302)] = cor

    raj = [im for t, im in rajadas.items() if t >= rodada_minima] or list(rajadas.values())
    rodada = np.median(np.stack(raj), axis=0).astype(np.uint8)
    # a barra da rodada começa onde as duas medianas passam a diferir (abaixo do tabuleiro)
    dif = np.abs(rodada.astype(int) - fundo.astype(int)).max(axis=2)
    linhas = np.nonzero((dif[BY + cfg.GAME_REGION[3] + 5 :] > 40).mean(axis=1) > 0.05)[0]
    y_barra = BY + cfg.GAME_REGION[3] + 5 + int(linhas.min()) - 4
    return fundo, rodada[y_barra:], y_barra, cor


# ── robô e bola ─────────────────────────────────────────────────────────────
def extrair_robo(telas: dict, fundo: np.ndarray, bolas: dict):
    """
    Robô segurando a bola, alinhado pela posição da bola (o robô fica sempre
    à direita dela). Mediana dos recortes alinhados de rodadas com a bola
    longe das paredes. O texto "xN" é desenhado à parte (glifos do contador).
    """
    caixa = (cfg.px(-20), cfg.px(-80), cfg.px(110), cfg.px(100))  # x0, y0, x1, y1 relativos ao centro da bola
    recs, recs_f = [], []
    for t, (bx, by) in bolas.items():
        if not (cfg.px(40) < bx < cfg.px(330)) or t not in telas:
            continue
        cx, cy = BX + bx, BY + by
        y0, y1, x0, x1 = cy + caixa[1], cy + caixa[3], cx + caixa[0], cx + caixa[2]
        if y1 > fundo.shape[0]:
            continue
        recs.append(telas[t][y0:y1, x0:x1].astype(np.int16))
        recs_f.append(fundo[y0:y1, x0:x1].astype(np.int16))
    img = np.median(np.stack(recs), axis=0).astype(np.uint8)
    f = np.median(np.stack(recs_f), axis=0).astype(np.uint8)
    sprite = _rgba(img, f)
    # tira o texto "xN" (embaixo à esquerda da bola) do sprite; a bola na mão
    # fica: entre rodadas a tela mostra o robô segurando a bola
    ox, oy = -caixa[0], -caixa[1]
    sprite[oy + cfg.px(8) : oy + cfg.px(42), : ox + cfg.px(14), 3] = 0
    return sprite, (-ox, -oy)


def extrair_bola_voando(pasta: Path):
    """
    Bola em voo: recorte de uma bola isolada (componente branco redondo, longe
    de outros) nos quadros das rajadas, mediana de vários. Devolve BGRA e o
    deslocamento do canto em relação ao centro.
    """
    r = cfg.px(12)
    recs = []
    for q in sorted((pasta / "referencia").glob("rajada_t*/f_*.png"))[::7]:
        img = cv2.imread(str(q))
        m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cfg.BALL_HSV_LOW, cfg.BALL_HSV_HIGH)
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not (cfg.px(16) <= w <= cfg.px(20) and cfg.px(16) <= h <= cfg.px(20)):
                continue
            cx, cy = int(round(cen[i][0])), int(round(cen[i][1]))
            if cy < r + 2 or cx < r + 2 or cx > img.shape[1] - r - 3 or cy > cfg.px(540):
                continue
            rec = img[cy - r : cy + r + 1, cx - r : cx + r + 1]
            # só bolas sozinhas: nada claro além do círculo dentro do recorte
            if (cv2.inRange(cv2.cvtColor(rec, cv2.COLOR_BGR2HSV), (0, 0, 120), (179, 255, 255)) > 0).sum() > a * 1.3:
                continue
            recs.append(rec)
        if len(recs) >= 60:
            break
    img = np.median(np.stack(recs), axis=0).astype(np.uint8)
    fundo = np.full_like(img, img[0, 0])
    return _rgba(img, fundo), (-r, -r), len(recs)


# ── glifos ──────────────────────────────────────────────────────────────────
def _componentes(alfa: np.ndarray, limiar: float, h_min: int):
    m = (alfa > limiar).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return sorted([tuple(int(x) for x in s) for s in st[1:] if s[3] >= h_min], key=lambda s: s[0])


def extrair_glifos_hp(telas: dict):
    """
    Dígitos do HP: blocos quadrados roxos recém-nascidos (HP = rodada, sem
    marcador, rodada >= 6 pra ser roxo). Tinta preta -> alfa. Devolve
    {dígito: alfa}, o vão médio entre dígitos e a cor roxa.
    """
    por_d = collections.defaultdict(list)
    vaos = []
    roxo = None
    for t, tela in telas.items():
        if t < 6:
            continue
        board = tela[BY : BY + cfg.GAME_REGION[3], BX : BX + cfg.GAME_REGION[2]]
        for b in ac.analisar_tabuleiro(board, t)["blocos"]:
            if b["tipo"] != "Q" or b["fileira"] != 1 or b["verde"] or b["marcador"]:
                continue
            x0, y0 = cfg.COL_LEFTS[b["coluna"]], ac.TOPO_0 + PASSO
            cel = board[y0 : y0 + cfg.px(60), x0 : x0 + cfg.px(60)]
            cor = np.median(cel.reshape(-1, 3), axis=0)
            if cor[0] < 200:  # não é roxo
                continue
            roxo = cor if roxo is None else roxo
            a = _alfa_preto(cel, cor)
            a[:3], a[-3:], a[:, :3], a[:, -3:] = 0, 0, 0, 0
            comps = _componentes(a, 0.5, cfg.px(10))
            txt = str(t)
            if len(comps) != len(txt):
                continue
            for (x, y, w, h, _), d in zip(comps, txt):
                por_d[d].append(a[y - 1 : y + h + 1, x - 1 : x + w + 1])
            for c1, c2 in zip(comps, comps[1:]):
                vaos.append(c2[0] - (c1[0] + c1[2]))
    glifos = {d: _media_alinhada(rs) for d, rs in por_d.items()}
    return glifos, float(np.median(vaos)) if vaos else 3.0, roxo


def extrair_glifos_linha(telas: dict, cor_fundo):
    """Dígitos brancos do "LINE N" (rótulo = número do turno, já conferido pelo analisar_capturas)."""
    por_d = collections.defaultdict(list)
    centros = []
    for t, tela in telas.items():
        reg = tela[BY - cfg.px(56) : BY - cfg.px(6), BX + cfg.px(150) : BX + cfg.px(302)]
        a = _alfa_branco(reg, cor_fundo)
        comps = _componentes(a, 0.5, cfg.px(30))
        txt = str(t)
        if len(comps) != len(txt):
            continue
        for (x, y, w, h, _), d in zip(comps, txt):
            por_d[d].append(a[y - 1 : y + h + 1, x - 1 : x + w + 1])
        xa, xb = comps[0][0], comps[-1][0] + comps[-1][2]
        centros.append((cfg.px(150) + (xa + xb) / 2, comps[0][1] - cfg.px(56)))
    glifos = {d: _media_alinhada(rs) for d, rs in por_d.items()}
    cx = float(np.median([c[0] for c in centros]))
    topo = float(np.median([c[1] for c in centros]))
    return glifos, cx, topo


def extrair_glifos_contador(telas: dict, bolas: dict, contagens: dict, cor_fundo):
    """
    "xN" embaixo à esquerda da bola. O rótulo de cada dígito vem do OCR do
    contador (vision.ler_contador, 100% nas sessões de teste). Devolve os
    glifos (inclusive o "x"), o deslocamento do canto do "x" em relação ao
    centro da bola e o vão entre caracteres.
    """
    por_d = collections.defaultdict(list)
    offs, vaos = [], []
    bx0, by0, bw, bh = cfg.BALL_COUNT_REGION
    gx, gy = cfg.GAME_REGION[:2]
    for t, tela in telas.items():
        if t not in bolas or t not in contagens:
            continue
        x0, y0 = BX + (bx0 - gx), BY + (by0 - gy)
        reg = tela[y0 : y0 + bh, x0 : x0 + bw]
        n = contagens[t]
        if v.ler_contador(reg) != n:
            continue
        a = _alfa_branco(reg, cor_fundo)
        comps = _componentes(a, 0.5, cfg.MIN_BALL_COUNT_DIGIT_H)
        comps = [c for c in comps if c[4] >= cfg.MIN_BALL_COUNT_DIGIT_AREA * 0.5]
        txt = "x" + str(n)
        if len(comps) != len(txt):
            continue
        for (x, y, w, h, _), d in zip(comps, txt):
            por_d[d].append(a[max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w + 1])
        for c1, c2 in zip(comps, comps[1:]):
            vaos.append(c2[0] - (c1[0] + c1[2]))
        bxc, byc = bolas[t]
        offs.append((x0 + comps[0][0] - BX - bxc, y0 + min(c[1] for c in comps[1:]) - BY - byc))
    glifos = {d: _media_alinhada(rs) for d, rs in por_d.items()}
    off = (float(np.median([o[0] for o in offs])), float(np.median([o[1] for o in offs])))
    return glifos, off, float(np.median(vaos)) if vaos else 2.0


# ── power-ups e ícones dos blocos ───────────────────────────────────────────
def extrair_powerups(telas: dict, fundo: np.ndarray):
    """Sprites dos power-ups, mediana dos recortes centrados no que o vision achou (nome pelo ÍCONE)."""
    r = cfg.px(26)
    por = collections.defaultdict(list)
    for t, tela in telas.items():
        board = tela[BY : BY + cfg.GAME_REGION[3], BX : BX + cfg.GAME_REGION[2]]
        for p in v.detect_powerups(board):
            nome = {"laser_v": "LH", "laser_h": "LV", "extra_ball": "+", "bonus_b": "B", "random": "R"}.get(p.power_up_type.value)
            if nome is None:
                continue
            cx, cy = BX + int(round(p.cx)), BY + int(round(p.cy))
            # centro "verdadeiro" = centróide do círculo colorido (detect_powerups pode errar 1 px)
            por[nome].append((tela[cy - r : cy + r + 1, cx - r : cx + r + 1], fundo[cy - r : cy + r + 1, cx - r : cx + r + 1]))
    sprites = {}
    for nome, rs in por.items():
        rs = [x for x in rs if x[0].shape == (2 * r + 1, 2 * r + 1, 3)]
        img = np.median(np.stack([x[0] for x in rs]), axis=0).astype(np.uint8)
        f = np.median(np.stack([x[1] for x in rs]), axis=0).astype(np.uint8)
        sprites[nome] = _rgba(img, f)
    return sprites


def extrair_icones_bloco(telas: dict):
    """
    Corrente (Share Block) e contorno do divisor (mini bloco), como alfa de
    tinta preta num quadrado de ICONE_LADO px cujo canto superior direito fica
    ICONE_FORA px acima/à direita do canto do bloco, mais a máscara da parte do
    mini bloco que sai PARA FORA do bloco (pintada com a cor dele). O bloco é
    localizado pela mancha de cor real (não pela grade do config).
    """
    k, fora_px = ICONE_LADO, ICONE_FORA
    corr, div, div_fora = [], [], []
    for t, tela in telas.items():
        board = tela[BY : BY + cfg.GAME_REGION[3], BX : BX + cfg.GAME_REGION[2]]
        mask = ac._mascara_cor(board)
        for b in ac.analisar_tabuleiro(board, t)["blocos"]:
            if b["tipo"] != "Q" or not b["marcador"]:
                continue
            x0 = int(round(geo.x_coluna(b["coluna"])))
            y0 = int(round(geo.y_fileira(b["fileira"])))
            cel = board[y0 : y0 + 60, x0 : x0 + 60]
            cor = np.median(cel.reshape(-1, 3), axis=0)
            xb = x0 + int(round(geo.LADO)) + fora_px  # borda direita do quadrado do ícone
            ya = y0 - fora_px
            if ya < 0 or xb > board.shape[1]:
                continue
            canto = board[ya : ya + k, xb - k : xb]
            colorido = mask[ya : ya + k, xb - k : xb]
            # tinta só DENTRO da área colorida (bloco + mini bloco): o fechamento
            # morfológico preenche as linhas finas do contorno lá dentro sem
            # avançar pro fundo escuro de fora, que o alfa_preto confundiria com
            # tinta (antigo: dilate 3x3 — virava uma moldura preta)
            # (margem zerada: sem ela o fechamento enche a borda do recorte)
            pad = cv2.copyMakeBorder(colorido, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
            dentro = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))[3:-3, 3:-3] > 0
            a = _alfa_preto(canto, cor) * dentro
            (corr if b["marcador"] == "corrente" else div).append(a)
            if b["marcador"] == "divisor":
                div_fora.append(colorido.astype(np.float32))
    out = {}
    if corr:
        out["corrente"] = np.median(np.stack(corr), axis=0)
    if div:
        out["divisor"] = np.median(np.stack(div), axis=0)
        out["divisor_cor"] = np.median(np.stack(div_fora), axis=0)
    return out


def main():
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "logs" / "captura_analise_2"
    telas, rajadas, reg = _carregar(pasta)
    log = (pasta / "sessao.log").read_text(encoding="utf-8", errors="ignore")
    bolas, contagens = {}, {}
    for parte in re.split(r"===== TURNO ", log)[1:]:
        t = int(parte.split()[0])
        m = re.search(r"bola x=(\d+) y=(\d+)", parte)
        if m:
            bolas[t] = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"bolas \(OCR\): (\d+)", parte)
        if m:
            contagens[t] = int(m.group(1))

    SAIDA.mkdir(parents=True, exist_ok=True)
    fundo, barra_rodada, y_barra, cor_fundo = extrair_fundo(telas, rajadas, rodada_minima=31)
    cv2.imwrite(str(SAIDA / "fundo.png"), fundo)
    cv2.imwrite(str(SAIDA / "barra_rodada.png"), barra_rodada)

    robo, off_robo = extrair_robo(telas, fundo, bolas)
    cv2.imwrite(str(SAIDA / "robo.png"), robo)
    bola, off_bola, n_bolas = extrair_bola_voando(pasta)
    cv2.imwrite(str(SAIDA / "bola.png"), bola)

    g_hp, vao_hp, roxo = extrair_glifos_hp(telas)
    g_linha, cx_linha, topo_linha = extrair_glifos_linha(telas, cor_fundo)
    g_cont, off_cont, vao_cont = extrair_glifos_contador(telas, bolas, contagens, cor_fundo)
    for prefixo, gl in (("hp", g_hp), ("linha", g_linha), ("contador", g_cont)):
        for d, a in gl.items():
            cv2.imwrite(str(SAIDA / f"{prefixo}_{d}.png"), (a * 255).astype(np.uint8))

    pus = extrair_powerups(telas, fundo)
    nomes_pu = {"+": "mais", "LH": "laser_h", "LV": "laser_v", "R": "aleatorio", "B": "moeda"}
    for nome, s in pus.items():
        cv2.imwrite(str(SAIDA / f"pu_{nomes_pu[nome]}.png"), s)

    icones = extrair_icones_bloco(telas)
    for nome, a in icones.items():
        cv2.imwrite(str(SAIDA / f"icone_{nome}.png"), (a * 255).astype(np.uint8))

    medidas = {
        "origem": str(pasta.name),
        "janela": {"largura": int(fundo.shape[1]), "altura": int(fundo.shape[0]), "tabuleiro_x": BX, "tabuleiro_y": BY},
        "cor_fundo_bgr": [int(c) for c in cor_fundo],
        "y_barra_rodada": int(y_barra),
        "robo_offset": [int(o) for o in off_robo],
        "bola_offset": [int(o) for o in off_bola],
        "bola_amostras": n_bolas,
        "hp_vao": vao_hp,
        "hp_roxo_bgr": [int(c) for c in roxo],
        "linha_centro_x": cx_linha,
        "linha_topo_y": topo_linha,
        "contador_offset": list(off_cont),
        "contador_vao": vao_cont,
        "glifos_hp": sorted(g_hp),
        "glifos_linha": sorted(g_linha),
        "glifos_contador": sorted(g_cont),
        "powerups": sorted(nomes_pu[n] for n in pus),
        "icones": sorted(icones),
        "icone_lado": ICONE_LADO,
        "icone_fora": ICONE_FORA,
    }
    with open(SAIDA / "assets.json", "w", encoding="utf-8") as f:
        json.dump(medidas, f, indent=2, ensure_ascii=False)
    print(json.dumps(medidas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
