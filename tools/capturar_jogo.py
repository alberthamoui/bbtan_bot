"""
capturar_jogo.py — Captura telas LIMPAS do jogo real (sem nenhuma anotação do
bot), como referência para montar a versão local do BBTAN.

Dois jeitos de usar:

1. Manual (você joga com o mouse): este script fica escutando teclas globais
   (funciona com o foco no emulador):

     F8   captura única: a janela do jogo inteira (tabuleiro + HUD de cima e
          de baixo), em PNG. Use para a aparência: início do jogo, cada tipo
          de bloco e power-up, botões, tela de game over.
     F9   rajada: grava só o tabuleiro (GAME_REGION) na maior taxa possível
          por --segundos (padrão 4). Aperte F9 e DEPOIS dispare: serve para
          medir a velocidade da bola, o intervalo entre bolas e o efeito do
          acelerar.
     F10  sai.

   Saída em capturas/AAAAMMDD_HHMMSS/ (fora do git).

2. Pelo bot: `python main.py --referencia` usa as funções daqui pra salvar a
   tela antes de cada tiro e uma rajada a cada --rajada-cada turnos, dentro
   da pasta do log da sessão (logs/captura_analise_N/referencia/).

Arquivos:
  tela_*.png               janela do jogo inteira
  rajada_*/f_NNNN.png      quadros da rajada (só o GAME_REGION)
  rajada_*/tempos.csv      quadro,segundos desde o início da rajada
  rajada_*/eventos.json    instantes marcados (disparo, acelerar...), no mesmo relógio
  regioes.json             onde na tela cada captura foi feita (pra alinhar
                           com GAME_REGION depois)

Uso manual:
  python tools/capturar_jogo.py
  python tools/capturar_jogo.py --segundos 6
"""

import argparse
import ctypes
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import mss
import numpy as np

import config as cfg

VK_F8, VK_F9, VK_F10 = 0x77, 0x78, 0x79

# Margem em volta do GAME_REGION na captura da janela, no perfil de referência
# (452 px de largura). Medido no logs/captura_analise_1: a janela do jogo vai
# de ~50 px à esquerda/direita do tabuleiro, o "LINE N" fica ~60 px acima
# dele e a barra de itens de baixo termina ~772 px abaixo do topo dele.
MARGEM_X = 60
MARGEM_CIMA = 160
# antigo: 820 — pegava a barra de tarefas do Windows embaixo da janela
MARGEM_BAIXO = 776  # medido do topo do GAME_REGION

PASTA_BASE = Path(__file__).resolve().parent.parent / "capturas"


def regiao_janela(sct) -> dict:
    """Retângulo da janela do jogo: GAME_REGION com margens, cortado aos limites dos monitores."""
    gx, gy, gw, _ = cfg.GAME_REGION
    tela = sct.monitors[0]  # todos os monitores juntos
    x0 = max(tela["left"], gx - cfg.px(MARGEM_X))
    y0 = max(tela["top"], gy - cfg.px(MARGEM_CIMA))
    x1 = min(tela["left"] + tela["width"], gx + gw + cfg.px(MARGEM_X))
    y1 = min(tela["top"] + tela["height"], gy + cfg.px(MARGEM_BAIXO))
    return {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}


def regiao_tabuleiro() -> dict:
    gx, gy, gw, gh = cfg.GAME_REGION
    return {"left": gx, "top": gy, "width": gw, "height": gh}


def _grab(sct, mon: dict) -> np.ndarray:
    return np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]


def salvar_janela(sct, pasta: Path, nome: str, regioes: dict) -> None:
    """Salva a janela do jogo inteira em pasta/nome e anota a região em `regioes`."""
    mon = regiao_janela(sct)
    cv2.imwrite(str(pasta / nome), _grab(sct, mon))
    regioes[nome] = mon


def salvar_regioes(pasta: Path, regioes: dict) -> None:
    dados = {"GAME_REGION": regiao_tabuleiro(), **regioes}
    with open(pasta / "regioes.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2)


class Gravador:
    """
    Grava o GAME_REGION numa thread, o mais rápido possível, por `segundos`.
    `marcar(nome)` anota um instante no mesmo relógio dos quadros (ex: o
    disparo e o clique no acelerar). Se `janela_em` for dado, também tira UMA
    foto da janela inteira nesse instante (o HUD durante a rodada).
    """

    def __init__(self, segundos: float, janela_em: Optional[float] = None):
        self.segundos = segundos
        self.janela_em = janela_em
        self.quadros: list = []
        self.tempos: list = []
        self.eventos: dict = {}
        self.janela: Optional[np.ndarray] = None
        self.janela_regiao: Optional[dict] = None
        self._t0 = 0.0
        self._thread: Optional[threading.Thread] = None

    def iniciar(self) -> None:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._rodar, daemon=True)
        self._thread.start()

    def marcar(self, nome: str) -> None:
        self.eventos[nome] = round(time.perf_counter() - self._t0, 5)

    def esperar(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _rodar(self) -> None:
        mon = regiao_tabuleiro()
        with mss.mss() as sct:  # mss não pode ser compartilhado entre threads
            while (t := time.perf_counter() - self._t0) < self.segundos:
                self.quadros.append(_grab(sct, mon))
                self.tempos.append(t)
                if self.janela_em is not None and self.janela is None and t >= self.janela_em:
                    self.janela_regiao = regiao_janela(sct)
                    self.janela = _grab(sct, self.janela_regiao)
                    self.eventos["foto_janela"] = round(time.perf_counter() - self._t0, 5)

    def fps(self) -> float:
        return len(self.tempos) / max(self.tempos[-1], 1e-6) if self.tempos else 0.0

    def salvar(self, pasta: Path, nome: str, regioes: dict) -> Path:
        """Grava os quadros em pasta/nome/ (PNG sem perda), tempos.csv e eventos.json."""
        sub = pasta / nome
        sub.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(self.quadros):
            # compressão 1: PNG continua sem perda, só grava mais rápido
            cv2.imwrite(str(sub / f"f_{i:04d}.png"), q, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        with open(sub / "tempos.csv", "w", encoding="utf-8") as f:
            f.write("quadro,segundos\n")
            f.writelines(f"{i},{t:.5f}\n" for i, t in enumerate(self.tempos))
        with open(sub / "eventos.json", "w", encoding="utf-8") as f:
            json.dump(self.eventos, f, indent=2)
        regioes[nome] = regiao_tabuleiro()
        if self.janela is not None:
            cv2.imwrite(str(pasta / f"tela_{nome}.png"), self.janela)
            regioes[f"tela_{nome}.png"] = self.janela_regiao
        return sub


def _apertou(vk: int, estado: dict) -> bool:
    """True só na borda de descida da tecla (segurar não repete)."""
    agora = bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    antes = estado.get(vk, False)
    estado[vk] = agora
    return agora and not antes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segundos", type=float, default=4.0, help="duração da rajada do F9 (padrão 4)")
    args = ap.parse_args()

    pasta = PASTA_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta.mkdir(parents=True)
    regioes: dict = {}
    print(f"Salvando em {pasta}")
    print("F8 = captura da tela | F9 = rajada do tabuleiro | F10 = sair")

    estado: dict = {}
    n_tela = n_rajada = 0
    with mss.mss() as sct:
        # primeira leitura só zera as bordas (tecla já apertada não dispara)
        for vk in (VK_F8, VK_F9, VK_F10):
            _apertou(vk, estado)
        try:
            while True:
                if _apertou(VK_F10, estado):
                    break
                if _apertou(VK_F8, estado):
                    n_tela += 1
                    nome = f"tela_{n_tela:03d}.png"
                    salvar_janela(sct, pasta, nome, regioes)
                    salvar_regioes(pasta, regioes)
                    print(f"  [F8] {nome}")
                if _apertou(VK_F9, estado):
                    n_rajada += 1
                    print(f"  [F9] gravando {args.segundos:.0f}s... dispare agora")
                    g = Gravador(args.segundos)
                    g.iniciar()
                    g.esperar()
                    sub = g.salvar(pasta, f"rajada_{n_rajada:03d}", regioes)
                    salvar_regioes(pasta, regioes)
                    print(f"  [F9] {sub.name}: {len(g.quadros)} quadros ({g.fps():.0f} fps)")
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
    salvar_regioes(pasta, regioes)
    print(f"\n{n_tela} capturas e {n_rajada} rajadas em {pasta}")


if __name__ == "__main__":
    main()
