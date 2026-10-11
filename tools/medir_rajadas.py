"""
medir_rajadas.py — Mede a física do jogo REAL nas rajadas gravadas
(logs/captura_analise_N/referencia/rajada_*), sem usar nada do physics.py.

Pra cada quadro acha as bolas (círculos brancos, centro subpixel) e guarda em
rajada_*/bolas.json. Depois mede, só com essas posições:
  - quando a 1a bola sai, a velocidade antes/depois do acelerar e o intervalo
    entre bolas (espaçamento na reta de saída);
  - até onde o centro da bola chega nas paredes (esquerda, direita, topo);
  - onde a bola para no chão.

  python tools/medir_rajadas.py                 # todas as rajadas de logs/
  python tools/medir_rajadas.py --refazer       # recalcula os bolas.json
  python tools/medir_rajadas.py --rebotes       # onde a 1a bola reflete (paredes, faces)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
AREA_BOLA = 260.0  # px² de uma bola (raio ~9,1) na máscara min(BGR) >= 200
LIMIAR = 200


def achar_bolas(quadro: np.ndarray) -> list:
    """Centros (x, y) das bolas brancas. Bolas encostadas são separadas pelos
    picos da transformada de distância."""
    mn = quadro.min(axis=2)
    m = (mn >= LIMIAR).astype(np.uint8)
    n, rot, st, _ = cv2.connectedComponentsWithStats(m)
    peso = np.clip((mn.astype(np.float32) - 40) / 215, 0, 1)
    saida = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.45 * AREA_BOLA:
            continue
        k = int(round(a / AREA_BOLA))
        if k <= 1 and max(w, h) <= 24:
            centros = [None]
        else:
            # várias bolas grudadas: picos da distância (centro de cada uma)
            sub = (rot[y : y + h, x : x + w] == i).astype(np.uint8)
            d = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            pico = (d >= 6.0) & (d == cv2.dilate(d, np.ones((9, 9), np.uint8)))
            ys, xs = np.nonzero(pico)
            centros = []
            for px_, py_ in sorted(zip(xs, ys), key=lambda p: -d[p[1], p[0]]):
                if all((px_ - a_) ** 2 + (py_ - b_) ** 2 > 12**2 for a_, b_ in centros):
                    centros.append((px_, py_))
            centros = [(x + a_, y + b_) for a_, b_ in centros] or [None]
        for c in centros:
            if c is None:
                cx, cy = x + w / 2, y + h / 2
            else:
                cx, cy = c
            # centroide ponderado (borda antisserrilhada) numa janela de 23 px
            x0, y0 = int(max(0, round(cx) - 11)), int(max(0, round(cy) - 11))
            x1, y1 = int(min(quadro.shape[1], round(cx) + 12)), int(min(quadro.shape[0], round(cy) + 12))
            p = peso[y0:y1, x0:x1]
            if c is not None or k > 1:
                # não deixa a bola vizinha puxar o centro: só o disco de raio 10
                yy, xx = np.mgrid[y0:y1, x0:x1]
                p = p * (((xx - cx) ** 2 + (yy - cy) ** 2) <= 10.5**2)
            s = p.sum()
            if s < 0.3 * AREA_BOLA:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            saida.append((round(float((p * xx).sum() / s), 2), round(float((p * yy).sum() / s), 2)))
    return saida


def rajadas() -> list:
    return sorted(RAIZ.glob("logs/captura_analise_*/referencia/rajada_*"))


def turno_de(rajada: Path) -> int:
    return int(re.search(r"t(\d+)$", rajada.name).group(1))


def carregar(rajada: Path, refazer: bool = False) -> dict:
    """{"tempos": [...], "bolas": [[(x, y), ...] por quadro], "eventos": {...}}"""
    cache = rajada / "bolas.json"
    if cache.exists() and not refazer:
        return json.loads(cache.read_text(encoding="utf-8"))
    tempos = [float(l.split(",")[1]) for l in (rajada / "tempos.csv").read_text().splitlines()[1:]]
    bolas = [achar_bolas(cv2.imread(str(rajada / f"f_{i:04d}.png"))) for i in range(len(tempos))]
    dados = {"tempos": tempos, "bolas": bolas, "eventos": json.loads((rajada / "eventos.json").read_text())}
    cache.write_text(json.dumps(dados), encoding="utf-8")
    return dados


PASSO = 13.308  # px por tick de física (mediana de 54 trechos retos)
TICK_S = 0.02  # o jogo atualiza a 50 Hz: a cada 6 quadros de 60 fps, 1 se repete


def rastrear_lider(dados: dict, ate: float = None) -> tuple:
    """
    Segue a 1a bola, tick a tick, até o acelerar (depois dele o jogo pula
    3-4 ticks por quadro). Devolve ((x, y) de partida, [(quadro, t, (x, y), n)])
    com n = ticks desde o ponto anterior. Quadro repetido (o jogo não andou)
    é pulado; bola escondida atrás do robô ou de outra bola é reencontrada
    pela previsão em linha reta.
    """
    T, B, ev = dados["tempos"], dados["bolas"], dados["eventos"]
    ate = ate if ate is not None else ev["acelerar"] + 0.08
    sx, sy = next(b for b in B[0] if b[1] > 555)
    p = v = None
    seq, perdido, tol = [], 0, 4.0
    for i, (t, bs) in enumerate(zip(T, B)):
        if t > ate:
            break
        bs = [tuple(b) for b in bs if np.hypot(b[0] - sx, b[1] - sy) > 2.5]
        if p is None:
            # a 1a que aparece longe do robô (ele esconde a saída pra direita)
            c = sorted((b for b in bs if 20 <= np.hypot(b[0] - sx, b[1] - sy) <= 200 and b[1] < sy - 2),
                       key=lambda b: np.hypot(b[0] - sx, b[1] - sy))
            if c:
                b = c[0]
                dd = np.hypot(b[0] - sx, b[1] - sy)
                p, v = b, ((b[0] - sx) / dd * PASSO, (b[1] - sy) / dd * PASSO)
                seq.append((i, t, p, max(1, round(dd / PASSO))))
            continue
        if any(np.hypot(b[0] - p[0], b[1] - p[1]) < 0.3 for b in bs):
            continue  # quadro repetido
        melhor = None
        for n in range(1, 6 + perdido * 2):
            q = (p[0] + n * v[0], p[1] + n * v[1])
            for b in bs:
                dp = np.hypot(b[0] - p[0], b[1] - p[1])
                if dp > n * PASSO + 0.8 or dp < (n - 2) * PASSO:
                    continue
                e = np.hypot(b[0] - q[0], b[1] - q[1])
                if melhor is None or e < melhor[0]:
                    melhor = (e, n, b)
            if melhor and melhor[0] < 1.5:
                break
        if melhor is None or melhor[0] > 2 * PASSO:
            perdido += 1
            continue
        e, n, b = melhor
        if e < tol:
            v = ((b[0] - p[0]) / n, (b[1] - p[1]) / n)
        else:  # bateu: chuta a reflexão pelo eixo que mais explica o desvio
            q = (p[0] + n * v[0], p[1] + n * v[1])
            v = (-v[0], v[1]) if abs(b[0] - q[0]) > abs(b[1] - q[1]) else (v[0], -v[1])
        if len(seq) >= 3:
            tol = 1.5  # a direção inicial (saindo de trás do robô) é imprecisa
        perdido = 0
        p = b
        seq.append((i, t, p, n))
    return (sx, sy), seq


def segmentos(seq: list) -> list:
    """Trechos retos (velocidade constante por tick, resíduo < 0,9 px)."""
    K = np.cumsum([0] + [s[3] for s in seq[1:]])
    P = np.array([s[2] for s in seq], float)
    segs, a = [], 0

    def ajuste(idx):
        A = np.vstack([K[idx], np.ones(len(idx))]).T
        cx = np.linalg.lstsq(A, P[idx, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A, P[idx, 1], rcond=None)[0]
        return cx, cy, np.hypot(P[idx, 0] - A @ cx, P[idx, 1] - A @ cy).max()

    while a < len(P) - 1:
        b = a + 1
        while b + 1 < len(P) and ajuste(np.arange(a, b + 2))[2] <= 0.9:
            b += 1
        if b - a + 1 >= 3:
            cx, cy, _ = ajuste(np.arange(a, b + 1))
            segs.append({"k0": int(K[a]), "k1": int(K[b]), "vx": cx[0], "vy": cy[0], "x0": cx[1], "y0": cy[1],
                         "n": b - a + 1})
        a = b
    return segs


def rebotes(segs: list) -> list:
    """
    Reflexões limpas entre dois trechos (um eixo inverte, o outro mantém).
    Se a bola reflete no instante exato do contato, as duas retas são
    espelhadas na superfície que o CENTRO toca: s = (x0_antes + x0_depois) / 2.
    """
    out = []
    for A, B in zip(segs, segs[1:]):
        if B["k0"] - A["k1"] > 3 or abs(np.hypot(B["vx"], B["vy"]) - PASSO) > 0.2:
            continue
        if abs(B["vx"] + A["vx"]) < 0.4 and abs(B["vy"] - A["vy"]) < 0.4 and abs(A["vx"]) > 1:
            s = (A["x0"] + B["x0"]) / 2
            k = (s - A["x0"]) / A["vx"]
            out.append({"eixo": "x", "indo": "dir" if A["vx"] > 0 else "esq", "s": s,
                        "em": A["y0"] + A["vy"] * k, "tick": k})
        elif abs(B["vy"] + A["vy"]) < 0.4 and abs(B["vx"] - A["vx"]) < 0.4 and abs(A["vy"]) > 1:
            s = (A["y0"] + B["y0"]) / 2
            k = (s - A["y0"]) / A["vy"]
            out.append({"eixo": "y", "indo": "baixo" if A["vy"] > 0 else "cima", "s": s,
                        "em": A["x0"] + A["vx"] * k, "tick": k})
    return out


# limites do CENTRO da bola nas paredes (rebotes do lider: x-esq 10,97-11,12
# em 7 rebotes, x-dir 438,8-439,5 em 5, teto 12,66-12,80 em 3)
PAREDE_ESQ, PAREDE_DIR, TETO = 11.0, 439.3, 12.7


def _caminho_paredes(sx: float, sy: float, ang: float, comprimento: float = 1400.0) -> np.ndarray:
    """Polilinha da saída refletindo só nas paredes (sem blocos): [(x, y, s)]."""
    dx, dy = np.cos(np.radians(ang)), -np.sin(np.radians(ang))
    pts, x, y, s = [(sx, sy, 0.0)], sx, sy, 0.0
    while s < comprimento:
        ts = []
        if dx < 0:
            ts.append(((PAREDE_ESQ - x) / dx, "x"))
        if dx > 0:
            ts.append(((PAREDE_DIR - x) / dx, "x"))
        if dy < 0:
            ts.append(((TETO - y) / dy, "y"))
        t, eixo = min(ts) if ts else (comprimento, "")
        t = min(t, comprimento - s)
        x, y, s = x + dx * t, y + dy * t, s + t
        pts.append((x, y, s))
        if eixo == "x":
            dx = -dx
        elif eixo == "y":
            dy = -dy
        else:
            break
    return np.array(pts)


def _projetar(pts: np.ndarray, cam: np.ndarray):
    """Distância de cada ponto à polilinha e a posição s (px de caminho) do pé."""
    melhor_d = np.full(len(pts), np.inf)
    melhor_s = np.zeros(len(pts))
    for (ax, ay, sa), (bx, by, sb) in zip(cam, cam[1:]):
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        if L2 == 0:
            continue
        u = np.clip(((pts[:, 0] - ax) * ex + (pts[:, 1] - ay) * ey) / L2, 0, 1)
        d = np.hypot(ax + u * ex - pts[:, 0], ay + u * ey - pts[:, 1])
        m = d < melhor_d
        melhor_d[m], melhor_s[m] = d[m], (sa + u * (sb - sa))[m]
    return melhor_d, melhor_s


def medir_saida(dados: dict, janela_s: float = 0.9) -> Optional[dict]:
    """
    Ângulo e instante de saída da 1a bola, só com as detecções (não precisa
    seguir uma bola — a 1a costuma sair escondida atrás do robô).
      - ângulo: o que põe mais bolas detectadas (de todos os quadros do
        começo) em cima do caminho reto + paredes saindo da partida;
      - instante: em cada quadro, a bola mais adiantada no caminho é a 1a;
        t0 = t - s/(PASSO/TICK_S), pegando o envelope de baixo (as outras
        bolas vêm 3 ticks atrás e dão valores maiores).
    """
    T, B, ev = dados["tempos"], dados["bolas"], dados["eventos"]
    sx, sy = next(b for b in B[0] if b[1] > 555)
    ini = ev.get("swipe_fim", 0.0) - 0.05
    pts, tq = [], []
    for t, bs in zip(T, B):
        if ini <= t <= ini + janela_s:
            for x, y in bs:
                if np.hypot(x - sx, y - sy) > 20 and y < sy - 2:
                    pts.append((x, y))
                    tq.append(t)
    if len(pts) < 8:
        return None
    pts, tq = np.array(pts), np.array(tq)

    def nota(ang):
        d, _ = _projetar(pts, _caminho_paredes(sx, sy, ang))
        return (d < 1.5).sum() - 0.01 * np.minimum(d, 1.5).sum()

    grosso = max(np.arange(5.0, 175.0, 0.25), key=nota)
    ang = max(np.arange(grosso - 0.3, grosso + 0.3, 0.01), key=nota)
    d, s = _projetar(pts, _caminho_paredes(sx, sy, ang))
    ok = d < 1.5
    if ok.sum() < 6:
        return None
    # por quadro, a mais adiantada no caminho
    frente = {}
    for t, si in zip(tq[ok], s[ok]):
        frente[t] = max(frente.get(t, 0.0), si)
    v = np.array([t - si / PASSO * TICK_S for t, si in frente.items() if si > 40])
    if len(v) < 3:
        return None
    base = v.min()
    t0 = float(np.median(v[v <= base + 0.03]))
    return {"angulo": float(ang), "t0": t0, "partida": (sx, sy), "inliers": int(ok.sum())}


def medir_rebotes() -> None:
    """Rebotes limpos da 1a bola, com o que ela tocou (parede ou face de bloco
    da captura antes do tiro) e a que distância da borda VISÍVEL o centro
    refletiu. É daqui que saem PAREDE_*, TETO e o contato com os blocos."""
    import cv2

    import analisar_capturas as ac
    from jogo_local import geometria as geo

    v = []
    for r in rajadas():
        d = carregar(r)
        _, seq = rastrear_lider(d)
        segs = segmentos(seq)
        v += [np.hypot(s["vx"], s["vy"]) for s in segs if s["n"] >= 5]
        t = turno_de(r)
        tela = cv2.imread(str([q for q in r.parent.glob(f"tela_*t{t:03d}.png") if "rajada" not in q.name][0]))
        blocos = ac.analisar_tabuleiro(tela[160:738, 60:512], t)["blocos"]
        for rb in rebotes(segs):
            s, em = rb["s"], rb["em"]
            cand = []
            for b in blocos:
                x0, y0 = geo.x_coluna(b["coluna"]), geo.y_fileira(b["fileira"])
                lado = geo.LADO_2X2 if b["tipo"] == "2x2" else geo.LADO
                nome = f"{b['tipo']} L{b['fileira']}C{b['coluna']}"
                if rb["eixo"] == "x" and y0 - 15 <= em <= y0 + lado + 15:
                    cand.append((nome + (" esq" if rb["indo"] == "dir" else " dir"),
                                 x0 - s if rb["indo"] == "dir" else s - x0 - lado))
                if rb["eixo"] == "y" and x0 - 15 <= em <= x0 + lado + 15:
                    cand.append((nome + (" topo" if rb["indo"] == "baixo" else " base"),
                                 y0 - s if rb["indo"] == "baixo" else s - y0 - lado))
            cand = [c for c in cand if -2 < c[1] < 30]
            alvo = min(cand, key=lambda c: abs(c[1] - 7))[0] if cand else ""
            dist = min(cand, key=lambda c: abs(c[1] - 7))[1] if cand else float("nan")
            if s < 30 or (rb["eixo"] == "x" and s > 420):
                o_que = "parede" if rb["eixo"] == "x" else "teto"
            elif cand:
                o_que = f"{alvo}: {dist:.2f} px da borda visivel"
            else:
                o_que = "? (nada do tabuleiro ali: provavel troca de bola no rastreio)"
            print(f"{r.name:16s} {rb['eixo']}-{rb['indo']:5s} centro refletiu em {s:7.2f} (no outro eixo {em:6.1f}) -> {o_que}")
    print(f"passo por tick: mediana {np.median(v):.3f} px (p5 {np.percentile(v, 5):.3f}, p95 {np.percentile(v, 95):.3f}, "
          f"{len(v)} trechos retos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--rebotes", action="store_true")
    args = ap.parse_args()
    if args.rebotes:
        sys.path.append(str(Path(__file__).resolve().parent))
        medir_rebotes()
        return
    for r in rajadas():
        d = carregar(r, args.refazer)
        n = [len(b) for b in d["bolas"]]
        s = medir_saida(d)
        extra = f" | saida {s['angulo']:.2f} graus, t0 {s['t0']:.3f}s" if s else ""
        print(f"{r.parent.parent.name}/{r.name}: {len(n)} quadros, max {max(n)} bolas num quadro{extra}")


if __name__ == "__main__":
    main()
