"""
analisar_capturas.py — Lê as capturas limpas do modo --referencia
(logs/captura_analise_N/referencia/tela_*.png) e extrai, rodada a rodada:

  - a rodada ("LINE N" no topo da tela), por OCR com exemplos reais
    (templates/linha/);
  - o tabuleiro inteiro: quantidade e tipos de blocos e de power-ups;
  - a FILEIRA NOVA (a que acabou de nascer no topo): blocos, tipo, HP em
    relação à rodada (1x, 2x, 3x), marcadores (divisor / corrente) e
    power-ups. É ela que mostra como o jogo gera as fileiras.

Classifica cada célula pelos pixels (não depende do detect_blocks do bot):
  - cada bloco é uma mancha colorida (o vão escuro separa os vizinhos; um
    2x2 é uma mancha só). Fileira 0 = ACIMA da grade visível (y 3-63), onde o
    2x2 recém-nascido põe a metade de cima; 1-7 são L1-L7 do log do bot;
  - caixa de 1 célula cheia (> 70%) = quadrado, meio cheia = triângulo (o
    canto vazio diz a orientação); caixa de 2x2 células = bloco 2x2;
  - HP pelo OCR do bot (vision._read_hp);
  - marcador no canto superior direito: contorno alto em L = divisor
    (mini bloco); ícone baixo e largo = corrente ("Share Block": os blocos
    ligados dividem o HP, aviso do jogo na rodada 58 do captura_analise_2);
  - verde = HP dobrado (as cores dos blocos normais variam com o HP);
  - power-ups pelo vision.detect_powerups. ATENÇÃO: lá os lasers estão
    trocados (o ◀▶ sai como laser_v); aqui o nome segue o ÍCONE: LH = ◀▶
    (feixe na fileira), LV = ▲▼ (feixe na coluna), confirmado na rajada da
    rodada 31 do captura_analise_2.

Saída, por pasta: analise_rodadas.csv, analise_fileira_nova.png (a faixa de
cima de cada rodada com a classificação escrita por cima, pra conferir a
olho) e, no terminal, o resumo por faixa de rodadas de todas as pastas.

Uso:
  python tools/analisar_capturas.py                       # todas as captura_analise_*
  python tools/analisar_capturas.py logs/captura_analise_2
"""

import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import config as cfg
import vision as v

RAIZ = Path(__file__).resolve().parent.parent
FILEIRAS = 8  # 0 = acima da grade (metade de cima do 2x2 novo), 1-7 = L1-L7
TOPO_0 = cfg.px(3)
PASSO = cfg.BLOCK_SIZE  # 64 px

# ── OCR da rodada ("LINE N") ────────────────────────────────────────────────
_LINHA_EX: dict = {}
_LINHA_H, _LINHA_CEL = 39, 30


def _canvas_linha(c: np.ndarray) -> np.ndarray:
    o = np.zeros((_LINHA_H + 2, _LINHA_CEL), dtype=bool)
    h, w = min(_LINHA_H, c.shape[0]), min(_LINHA_CEL - 2, c.shape[1])
    o[1 : 1 + h, 1 : 1 + w] = c[:h, :w] > 0
    return o


def _carregar_linha():
    if _LINHA_EX:
        return
    for p in sorted((RAIZ / "templates" / "linha").glob("*.png")):
        tira = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        ex = []
        for x0 in range(0, tira.shape[1], _LINHA_CEL):
            cel = tira[:, x0 : x0 + _LINHA_CEL] > 127
            ys, xs = np.nonzero(cel)
            if len(xs):
                ex.append(_canvas_linha(cel[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]))
        _LINHA_EX[p.stem] = np.array(ex)


def _melhor_digito(c: np.ndarray) -> str:
    a = _canvas_linha(c)
    melhor, nota_m = "?", 0.0
    for d, ex in _LINHA_EX.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s = np.roll(np.roll(ex, dy, 1), dx, 2)
                n = 1 - float(((s ^ a).sum((1, 2)) / np.maximum(1, s.sum((1, 2)) + a.sum())).min())
                if n > nota_m:
                    melhor, nota_m = d, n
    return melhor


def ler_rodada(tela: np.ndarray, x0: int, y0: int) -> int:
    """'LINE N' fica centralizado ~48 px acima do tabuleiro (x0, y0 = canto do tabuleiro na tela)."""
    _carregar_linha()
    regiao = tela[max(0, y0 - cfg.px(56)) : y0 - cfg.px(6), x0 + cfg.px(150) : x0 + cfg.px(302)]
    m = cv2.inRange(cv2.cvtColor(regiao, cv2.COLOR_BGR2HSV), (0, 0, 200), (179, 40, 255))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    comps = sorted([s for s in st[1:] if s[3] >= cfg.px(30)], key=lambda s: s[0])
    digitos = []
    for x, y, w, h, _a in comps:
        crop = m[y : y + h, x : x + w]
        if w >= cfg.px(40):  # dois dígitos encostados (visto na rodada 58): parte ao meio
            digitos += [crop[:, : w // 2], crop[:, w // 2 :]]
        else:
            digitos.append(crop)
    txt = "".join(_melhor_digito(c) for c in digitos)
    return int(txt) if txt.isdigit() else -1


# ── Blocos do tabuleiro ─────────────────────────────────────────────────────
def _mascara_cor(board: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 110)).astype(np.uint8)


def _marcador(board: np.ndarray, x: int, y: int, w: int) -> str:
    """
    Canto superior direito de um bloco quadrado (x, y, w = caixa do bloco):
    'divisor' = contorno escuro alto em L (o mini bloco), 'corrente' = ícone
    escuro baixo e largo (Share Block), '' = nada. As linhas olhadas (2 a 17
    px do topo) ficam acima do número do HP.
    """
    k = cfg.px(22)
    reg = board[y + cfg.px(2) : y + cfg.px(18), x + w - k : x + w - cfg.px(2)]
    escuro = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)[:, :, 2] < 80
    ys, xs = np.nonzero(escuro)
    if len(xs) < 8:
        return ""
    return "divisor" if ys.max() - ys.min() + 1 >= cfg.px(12) else "corrente"


def analisar_tabuleiro(board: np.ndarray, rodada: int) -> dict:
    """
    Cada bloco é uma mancha colorida separada das vizinhas pelo vão escuro
    (um 2x2 é uma mancha só). Tamanho da caixa em células (64 px) diz 1x1 ou
    2x2; o preenchimento diz quadrado (> 70%) ou triângulo (o canto vazio diz
    a orientação). Power-ups (círculos de ~30 px) e textos ficam de fora pelo
    tamanho mínimo da caixa.
    """
    cor = _mascara_cor(board)
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    n, rot, st, _ = cv2.connectedComponentsWithStats(cor, connectivity=4)
    minimo = cfg.px(40)
    blocos = []
    for i in range(1, n):
        x, y, w, h, area = (int(a) for a in st[i])
        if w < minimo or h < minimo:
            continue
        mask = rot[y : y + h, x : x + w] == i
        cols = max(1, round((w + cfg.px(4)) / PASSO))
        fils = max(1, round((h + cfg.px(4)) / PASSO))
        cheio = area / float(w * h)
        if cols == 1 and fils == 1:
            if cheio > 0.7:
                tipo = "Q"
            else:
                k = cfg.px(10)
                cantos = {
                    "tl": mask[:k, :k].mean(), "tr": mask[:k, -k:].mean(),
                    "bl": mask[-k:, :k].mean(), "br": mask[-k:, -k:].mean(),
                }
                vazio = min(cantos, key=cantos.get)
                # canto do corpo (ângulo reto) = oposto ao vazio, como no log do bot
                tipo = "T" + {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}[vazio]
        elif cols == 2 and fils == 2 and cheio > 0.7:
            tipo = "2x2"
        else:
            tipo = f"{cols}x{fils}?"
        tri = tipo.startswith("T")
        hp, _ = v._read_hp(board, x, y, w, h, is_triangle=tri, is_2x2=tipo == "2x2")
        matiz = float(np.median(hsv[y : y + h, x : x + w, 0][mask]))
        blocos.append({
            "fileira": int(round((y - TOPO_0) / PASSO)),
            "coluna": v._get_col_index(x + cfg.px(30)),
            "tipo": tipo, "hp": hp,
            "mult": round(hp / rodada, 2) if rodada > 0 and hp > 0 else None,
            "marcador": _marcador(board, x, y, w) if tipo == "Q" else "",
            "verde": 45 <= matiz <= 85, "matiz": matiz, "fils": fils,
        })

    pus = []
    for p in v.detect_powerups(board):
        f = int(round((p.cy - TOPO_0 - PASSO / 2) / PASSO))
        nome = {"laser_v": "LH", "laser_h": "LV", "extra_ball": "+", "bonus_b": "B", "random": "R"}.get(
            p.power_up_type.value, p.power_up_type.value
        )
        pus.append({"fileira": f, "coluna": v._get_col_index(p.cx), "tipo": nome})
    return {"blocos": blocos, "powerups": pus}


def _rotulo(b: dict) -> str:
    s = b["tipo"]
    if b["marcador"]:
        s += "-div" if b["marcador"] == "divisor" else "-cor"
    if b["verde"]:
        s += "-verde"
    return s


def analisar_pasta(pasta: Path) -> list:
    ref = pasta / "referencia"
    reg = json.load(open(ref / "regioes.json", encoding="utf-8"))
    g = reg["GAME_REGION"]
    linhas, faixas = [], []
    telas = sorted(p for p in ref.glob("tela_*.png") if "rajada" not in p.name)
    for p in telas:
        r = reg.get(p.name)
        if r is None:
            continue
        tela = cv2.imread(str(p))
        x0, y0 = g["left"] - r["left"], g["top"] - r["top"]
        board = tela[y0 : y0 + g["height"], x0 : x0 + g["width"]]
        turno = int(re.search(r"t(\d+)\.png", p.name).group(1))
        lida = ler_rodada(tela, x0, y0)
        # A rodada sobe de 1 em 1: leitura fora da sequência (ex: aviso do jogo
        # cobrindo o "LINE N" na rodada 58) usa a anterior + 1.
        rodada = lida
        if linhas and lida != linhas[-1]["rodada"] + 1:
            rodada = linhas[-1]["rodada"] + 1
        a = analisar_tabuleiro(board, rodada)
        # fileira nova: blocos que começam na fileira 0 (2x2 novo) ou cabem só na 1
        novos = [b for b in a["blocos"] if b["fileira"] == 0 or (b["fileira"] == 1 and b["fils"] == 1)]
        pus_novos = [u for u in a["powerups"] if u["fileira"] == 1]
        linhas.append({
            "pasta": pasta.name, "turno": turno, "rodada": rodada, "rodada_lida": lida,
            "blocos_tab": len(a["blocos"]),
            "tipos_tab": dict(collections.Counter(_rotulo(b) for b in a["blocos"])),
            "powerups_tab": len(a["powerups"]),
            "tipos_pu_tab": dict(collections.Counter(u["tipo"] for u in a["powerups"])),
            "novos": [(b["coluna"], _rotulo(b), b["hp"], b["mult"]) for b in sorted(novos, key=lambda b: b["coluna"])],
            "pu_novos": [(u["coluna"], u["tipo"]) for u in sorted(pus_novos, key=lambda u: u["coluna"])],
        })
        faixa = board[0 : TOPO_0 + 2 * PASSO].copy()
        for b in novos:
            x, y = cfg.COL_LEFTS[b["coluna"]], TOPO_0 + PASSO
            cv2.putText(faixa, _rotulo(b), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3)
            cv2.putText(faixa, _rotulo(b), (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
            cv2.putText(faixa, f"{b['hp']}", (x, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        for u in pus_novos:
            x, y = cfg.COL_LEFTS[u["coluna"]], TOPO_0 + PASSO
            cv2.putText(faixa, u["tipo"], (x + 20, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(faixa, f"rodada {rodada} (t{turno})", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        faixas.append(faixa)

    with open(pasta / "analise_rodadas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    if faixas:
        while len(faixas) % 3:
            faixas.append(np.zeros_like(faixas[0]))
        cv2.imwrite(
            str(pasta / "analise_fileira_nova.png"),
            np.vstack([np.hstack(faixas[i : i + 3]) for i in range(0, len(faixas), 3)]),
        )
    return linhas


def _categoria_hp(tipo: str) -> str:
    """
    HP da fileira nova em múltiplos da rodada, pelo tipo/cor (o OCR de HP do
    bot troca 6 por 5 e perde o "1" do 2x2): 2x2 = 3x, verde ou corrente = 2x,
    o resto = 1x. analise_rodadas.csv guarda o HP lido pra conferir.
    """
    if tipo.startswith("2x2"):
        return "3x"
    if "verde" in tipo or "-cor" in tipo:
        return "2x"
    return "1x"


def resumo(linhas: list) -> str:
    faixa_de = lambda r: (r - 1) // 10 * 10 + 1
    por = collections.defaultdict(list)
    for l in linhas:
        if l["rodada"] > 0:
            por[faixa_de(l["rodada"])].append(l)
    out = []
    out.append("FILEIRA NOVA por faixa de rodadas (Q = quadrado, T = triangulo, div = divisor, cor = corrente)")
    out.append("faixa   rodadas  blocos/fileira  colunas ocupadas  tipos de bloco                     HP          power-ups alem do +")
    for k in sorted(por):
        ls = por[k]
        n = [len(l["novos"]) for l in ls]
        ocup = [sum(2 if t.startswith("2x2") else 1 for _, t, _, _ in l["novos"]) + len(l["pu_novos"]) for l in ls]
        tipos = collections.Counter(
            ("T" if t.startswith("T") else t.split("-")[0]) + ("-div" if "-div" in t else "-cor" if "-cor" in t else "") + ("-verde" if "verde" in t else "")
            for l in ls for _, t, _, _ in l["novos"]
        )
        hp = collections.Counter(_categoria_hp(t) for l in ls for _, t, _, _ in l["novos"])
        extras = collections.Counter(t for l in ls for _, t in l["pu_novos"] if t != "+")
        out.append(
            f"{k:>3}-{k + 9:<3}  {len(ls):>5}    {np.mean(n):4.2f} ({min(n)}-{max(n)})     "
            f"{np.mean(ocup):4.2f} de 7        {dict(tipos)!s:<34} {dict(hp)!s:<11} {dict(extras)} em {len(ls)} fileiras"
        )
    mais = collections.Counter(sum(1 for _, t in l["pu_novos"] if t == "+") for l in linhas)
    out.append(f"'+' por fileira nova: {dict(sorted(mais.items()))} (quantidade de '+': numero de fileiras)")
    extras = collections.Counter(sum(1 for _, t in l["pu_novos"] if t != "+") for l in linhas)
    out.append(f"outros power-ups por fileira nova: {dict(sorted(extras.items()))}")
    tri = collections.Counter(t for l in linhas for _, t, _, _ in l["novos"] if t.startswith("T"))
    um = sum(1 for l in linhas for _, t, _, _ in l["novos"] if t.startswith(("Q", "T")))
    out.append(f"triangulos: {sum(tri.values())} de {um} blocos 1x1 ({100 * sum(tri.values()) / max(1, um):.0f}%), orientacoes {dict(tri)}")
    # conferência do OCR de HP contra a categoria
    ok = tot = 0
    for l in linhas:
        for _, t, hp, mult in l["novos"]:
            tot += 1
            ok += mult is not None and abs(mult - int(_categoria_hp(t)[0])) < 0.02
    out.append(f"HP lido pelo OCR bate com a categoria em {ok} de {tot} blocos novos (o resto e erro de OCR, ver analise_rodadas.csv)")
    primeira = {}
    for l in sorted(linhas, key=lambda l: l["rodada"]):
        for _, t, _, _ in l["novos"]:
            primeira.setdefault(t.replace("Ttl", "T").replace("Ttr", "T").replace("Tbl", "T").replace("Tbr", "T"), l["rodada"])
        for _, t in l["pu_novos"]:
            primeira.setdefault("pu " + t, l["rodada"])
    out.append("primeira rodada de cada tipo: " + ", ".join(f"{t}={r}" for t, r in sorted(primeira.items(), key=lambda x: x[1])))
    seq = [(l["pasta"], l["rodada"], len(l["novos"])) for l in sorted(linhas, key=lambda l: (l["pasta"], l["rodada"]))]
    pares = [(a[2], b[2]) for a, b in zip(seq, seq[1:]) if a[0] == b[0] and b[1] == a[1] + 1]
    if len(pares) > 2:
        x, y = np.array(pares).T
        out.append(f"correlacao entre blocos de uma fileira e da seguinte: {np.corrcoef(x, y)[0, 1]:+.2f} ({len(pares)} pares; negativa = muitos numa, poucos na proxima)")
    return "\n".join(out)


def main():
    pastas = [Path(a) for a in sys.argv[1:]] or sorted((RAIZ / "logs").glob("captura_analise_*"))
    todas = []
    for p in pastas:
        ls = analisar_pasta(p)
        fora = [(l["turno"], l["rodada_lida"]) for l in ls if l["rodada_lida"] != l["rodada"]]
        print(f"{p.name}: {len(ls)} rodadas | leitura do LINE fora da sequencia em {len(fora)} (turno, lido): {fora[:10]}")
        todas += ls
    print()
    print(resumo(todas))


if __name__ == "__main__":
    main()
