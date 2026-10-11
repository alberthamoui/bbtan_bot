"""
partidas_locais.py — Joga várias partidas do jogo local em paralelo, com o
relógio virtual, e resume até onde o bot chegou em cada uma.

Cada partida é um `python main.py --rapido --sem-janelas --semente S
--partidas 1` num processo próprio (um núcleo cada): a mesma semente sorteia
as mesmas fileiras, então dá pra comparar duas versões do bot nas MESMAS
partidas. O tempo de cada partida é quase todo a simulação de ângulos do bot.

  python tools/partidas_locais.py                    # sementes 1-8
  python tools/partidas_locais.py --sementes 1-20 --processos 10
  python tools/partidas_locais.py --nome antes       # guarda o resumo com um nome

Resumo em logs/partidas_locais/<nome ou data>.csv; os logs de cada partida
ficam em logs/local_sS_N/ como sempre.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PADRAO = re.compile(r"RESULTADO LOCAL: semente (\S+) \| partida \d+ \| rodada (\d+) \| bolas (\d+) \| turnos (\d+)")


def sementes(txt: str) -> list:
    out = []
    for parte in txt.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(parte))
    return out


def jogar(semente: int, extra: list, limite_s: float) -> dict:
    t0 = time.perf_counter()
    cmd = [sys.executable, "main.py", "--rapido", "--sem-janelas", "--semente", str(semente), "--partidas", "1"] + extra
    try:
        r = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="ignore",
                           timeout=limite_s, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        saida = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        saida = (e.stdout or b"").decode("utf-8", "ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
    m = PADRAO.search(saida)
    res = {"semente": semente, "segundos": round(time.perf_counter() - t0, 1)}
    if m:
        res.update(rodada=int(m.group(2)), bolas=int(m.group(3)), turnos=int(m.group(4)), status="game over")
    else:
        erro = "Traceback" in saida
        res.update(rodada="", bolas="", turnos="", status="erro" if erro else "sem game over (limite de tempo)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sementes", default="1-8", help="ex: 1-8 ou 1,3,5 (padrao 1-8)")
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limite-min", type=float, default=60.0, help="minutos por partida antes de desistir")
    ap.add_argument("--nome", default=None, help="nome do resumo (padrao: data e hora)")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="argumentos a mais pro main.py (depois de --)")
    args = ap.parse_args()
    extra = [a for a in args.extra if a != "--"]
    lista = sementes(args.sementes)

    print(f"{len(lista)} partidas, {args.processos} por vez" + (f" | main.py {' '.join(extra)}" if extra else ""))
    t0 = time.perf_counter()
    resultados = []
    with ThreadPoolExecutor(args.processos) as pool:
        for res in pool.map(lambda s: jogar(s, extra, args.limite_min * 60), lista):
            resultados.append(res)
            print(f"  semente {res['semente']:>3}: {res['status']:<10} rodada {res['rodada']!s:>3} | "
                  f"bolas {res['bolas']!s:>3} | {res['turnos']!s:>3} turnos | {res['segundos']:.0f}s")

    ok = [r for r in resultados if r["status"] == "game over"]
    if ok:
        rod = sorted(r["rodada"] for r in ok)
        media = sum(rod) / len(rod)
        print(f"\nrodada final: media {media:.1f} | mediana {rod[len(rod) // 2]} | min {rod[0]} | max {rod[-1]} "
              f"({len(ok)} de {len(resultados)} partidas) | {time.perf_counter() - t0:.0f}s no total")
    pasta = RAIZ / "logs" / "partidas_locais"
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f"{args.nome or datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(arq, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["semente", "status", "rodada", "bolas", "turnos", "segundos"])
        w.writeheader()
        w.writerows(resultados)
    print(f"resumo: {arq.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
