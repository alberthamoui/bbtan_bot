"""
gerador.py — Sorteia a fileira nova de cada rodada do jogo local, com as
frequências contadas nas capturas reais (assets/fileiras.json, gerado por
tools/estatisticas_fileiras.py).

Regras (contadas em 81 fileiras novas, rodadas 1-71):
  - toda fileira nova tem exatamente 1 "+" e, às vezes, mais 1 power-up;
  - quantos blocos e de que tipo depende da faixa de 10 rodadas;
  - HP: N (rodada) no quadrado, triângulo e divisor; 2N no verde e em cada
    bloco da corrente (que vem em par); 3N no 2x2;
  - o 2x2 ocupa a fileira nova e a de cima (fileira 0, fora da grade);
  - nenhum tipo aparece antes da rodada em que foi visto pela 1a vez.
Acima da última faixa com dados, repete a última com pelo menos 5 fileiras.
"""

import json
import random
from pathlib import Path
from typing import Optional

from jogo_local import geometria as geo
from jogo_local.estado import Bloco, Estado, PowerUp

ARQUIVO = Path(__file__).resolve().parent / "assets" / "fileiras.json"
MIN_FILEIRAS_FAIXA = 5


class Gerador:
    def __init__(self, rng: Optional[random.Random] = None, arquivo: Path = ARQUIVO):
        self.rng = rng or random.Random()
        self.dados = json.loads(arquivo.read_text(encoding="utf-8"))
        self.faixas = [f for f in self.dados["faixas"] if f["linhas"] >= MIN_FILEIRAS_FAIXA]
        self.primeira = self.dados["primeira_rodada"]
        self.orient = self.dados["orientacoes"]
        self.proxima_corrente = 1

    def faixa(self, rodada: int) -> dict:
        for f in self.faixas:
            if f["de"] <= rodada <= f["ate"]:
                return f
        return self.faixas[-1] if rodada > self.faixas[-1]["ate"] else self.faixas[0]

    def _sortear(self, contagem: dict) -> str:
        itens = [(k, v) for k, v in contagem.items() if v > 0]
        return self.rng.choices([k for k, _ in itens], weights=[v for _, v in itens])[0]

    def _pode(self, chave: str, rodada: int) -> bool:
        return rodada >= self.primeira.get(chave, 1)

    def nova_fileira(self, est: Estado) -> None:
        """Põe a fileira da rodada est.rodada na fileira 1 (e 0, pro 2x2)."""
        n_rod = est.rodada
        fx = self.faixa(n_rod)
        livres = list(range(geo.COLUNAS))
        self.rng.shuffle(livres)
        extra = None
        if self.rng.random() < fx["com_extra"] / max(1, fx["linhas"]):
            pus = {k: v for k, v in (fx["extras"] or self.dados["faixas"][-1]["extras"]).items() if self._pode("pu " + k, n_rod)}
            if pus:
                extra = self._sortear(pus)
        reservar = 1 + (extra is not None)  # o "+" e o extra precisam de coluna
        n = int(self._sortear(fx["n_blocos"]))
        n = min(n, geo.COLUNAS - reservar)
        tipos = {k: v for k, v in fx["tipos"].items() if self._pode(k, n_rod)}
        colocados = 0
        while colocados < n and len(livres) > reservar:
            t = self._sortear(tipos)
            if t == "2x2":
                par = self._vizinhas(livres)
                if par is None or len(livres) - 2 < reservar:
                    t = "Q"
                else:
                    for c in par:
                        livres.remove(c)
                    est.blocos.append(Bloco(coluna=min(par), fileira=0, tipo="2x2", hp=3 * n_rod, hp_base=3 * n_rod))
                    colocados += 1
                    continue
            if t == "cor":
                if colocados + 2 > n or len(livres) - 2 < reservar:
                    t = "Q"
                else:
                    cid = self.proxima_corrente
                    self.proxima_corrente += 1
                    for _ in range(2):
                        est.blocos.append(Bloco(coluna=livres.pop(), fileira=1, hp=2 * n_rod, hp_base=2 * n_rod, corrente=cid))
                    colocados += 2
                    continue
            c = livres.pop()
            if t == "T":
                est.blocos.append(Bloco(coluna=c, fileira=1, tipo=self._sortear(self.orient), hp=n_rod, hp_base=n_rod))
            elif t == "verde":
                est.blocos.append(Bloco(coluna=c, fileira=1, hp=2 * n_rod, hp_base=n_rod))
            elif t == "div":
                est.blocos.append(Bloco(coluna=c, fileira=1, hp=n_rod, hp_base=n_rod, divisor=True))
            else:
                est.blocos.append(Bloco(coluna=c, fileira=1, hp=n_rod, hp_base=n_rod))
            colocados += 1
        est.powerups.append(PowerUp(coluna=livres.pop(), fileira=1, tipo="+"))
        if extra is not None and livres:
            est.powerups.append(PowerUp(coluna=livres.pop(), fileira=1, tipo=extra))

    def _vizinhas(self, livres):
        pares = [(c, c + 1) for c in livres if c + 1 in livres]
        return self.rng.choice(pares) if pares else None
