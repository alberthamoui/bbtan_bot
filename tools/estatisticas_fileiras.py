"""
estatisticas_fileiras.py — Junta o que tools/analisar_capturas.py contou nas
capturas reais (analise_rodadas.csv de cada logs/captura_analise_N) num
resumo que o jogo local usa pra sortear a fileira nova de cada rodada:
jogo_local/assets/fileiras.json.

Por faixa de 10 rodadas: quantos blocos a fileira nova tem, de que tipo
(quadrado, triângulo, divisor, verde, corrente, 2x2) e quais power-ups além
do "+" (toda fileira tem exatamente 1 "+"). Fora das faixas medidas, o jogo
local repete a última.

  python tools/analisar_capturas.py logs/captura_analise_1 logs/captura_analise_2
  python tools/estatisticas_fileiras.py
"""

import ast
import collections
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "jogo_local" / "assets" / "fileiras.json"


def tipo_de(rotulo: str) -> str:
    if rotulo.startswith("T"):
        return "T"
    if rotulo == "2x2":
        return "2x2"
    return {"Q-div": "div", "Q-verde": "verde", "Q-cor": "cor"}.get(rotulo, "Q")


def main():
    arquivos = sorted(RAIZ.glob("logs/captura_analise_*/analise_rodadas.csv"))
    if not arquivos:
        sys.exit("nenhum analise_rodadas.csv — rode tools/analisar_capturas.py antes")
    faixas = collections.defaultdict(lambda: {
        "linhas": 0, "n_blocos": collections.Counter(), "tipos": collections.Counter(),
        "extras": collections.Counter(), "com_extra": 0,
    })
    orient = collections.Counter()
    primeira = {}
    for arq in arquivos:
        for r in csv.DictReader(open(arq, encoding="utf-8")):
            rod = int(r["rodada"])
            if rod <= 0:
                continue
            novos = ast.literal_eval(r["novos"])
            pus = ast.literal_eval(r["pu_novos"])
            f = faixas[(rod - 1) // 10 * 10 + 1]
            f["linhas"] += 1
            f["n_blocos"][len(novos)] += 1
            for _, rot, _, _ in novos:
                t = tipo_de(rot)
                f["tipos"][t] += 1
                primeira[t] = min(primeira.get(t, rod), rod)
                if t == "T":
                    orient[rot] += 1
            extras = [t for _, t in pus if t != "+"]
            f["com_extra"] += bool(extras)
            for t in extras:
                f["extras"][t] += 1
                primeira["pu " + t] = min(primeira.get("pu " + t, rod), rod)
    saida = {
        "fonte": [str(a.relative_to(RAIZ)).replace("\\", "/") for a in arquivos],
        "faixas": [
            {"de": de, "ate": de + 9, "linhas": f["linhas"], "n_blocos": {str(k): v for k, v in sorted(f["n_blocos"].items())},
             "tipos": dict(f["tipos"]), "extras": dict(f["extras"]), "com_extra": f["com_extra"]}
            for de, f in sorted(faixas.items())
        ],
        "orientacoes": dict(orient),
        "primeira_rodada": dict(sorted(primeira.items(), key=lambda x: x[1])),
    }
    SAIDA.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{SAIDA.relative_to(RAIZ)}: {sum(f['linhas'] for f in faixas.values())} fileiras de {len(arquivos)} sessoes")
    for fx in saida["faixas"]:
        print(f"  {fx['de']:>3}-{fx['ate']:<3} {fx['linhas']:>3} fileiras  blocos {fx['n_blocos']}  tipos {fx['tipos']}  extras {fx['extras']}")


if __name__ == "__main__":
    main()
