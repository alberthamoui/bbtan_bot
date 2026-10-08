"""
capturar_jogo.py — Captura telas LIMPAS do jogo real (sem nenhuma anotação do
bot), como referência para montar a versão local do BBTAN.

Deixe o emulador aberto e jogue normalmente com o mouse. Este script fica
escutando teclas globais (funciona com o foco no emulador):

  F8   captura única: a janela do jogo inteira (tabuleiro + HUD de cima e de
       baixo), em PNG. Use para a aparência: início do jogo, cada tipo de
       bloco e power-up, botões, tela de game over.
  F9   rajada: grava só o tabuleiro (GAME_REGION) na maior taxa possível por
       --segundos (padrão 4). Aperte F9 e DEPOIS dispare: serve para medir a
       velocidade da bola, o intervalo entre bolas e o efeito do acelerar.
  F10  sai.

Tudo vai para capturas/AAAAMMDD_HHMMSS/ (fora do git):
  tela_NNN.png             capturas do F8
  rajada_NNN/f_NNNN.png    quadros da rajada
  rajada_NNN/tempos.csv    quadro,segundos desde o início da rajada
  regioes.json             onde na tela cada captura foi feita (pra alinhar
                           com GAME_REGION depois)

Uso:
  python tools/capturar_jogo.py
  python tools/capturar_jogo.py --segundos 6
"""

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import mss
import numpy as np

import config as cfg

VK_F8, VK_F9, VK_F10 = 0x77, 0x78, 0x79

# Margem em volta do GAME_REGION na captura do F8, no perfil de referência
# (452 px de largura). O HUD de baixo vai até ~750 px abaixo do topo do
# tabuleiro (botão retrieve em 703-741); o de cima ainda não foi medido, por
# isso a sobra generosa. Se a janela sair cortada, aumente aqui.
MARGEM_X = 60
MARGEM_CIMA = 160
MARGEM_BAIXO = 820  # medido do topo do GAME_REGION

PASTA_BASE = Path(__file__).resolve().parent.parent / "capturas"


def _apertou(vk: int, estado: dict) -> bool:
    """True só na borda de descida da tecla (segurar não repete)."""
    agora = bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    antes = estado.get(vk, False)
    estado[vk] = agora
    return agora and not antes


def _regiao_janela(sct) -> dict:
    """Retângulo do F8: GAME_REGION com margens, cortado aos limites dos monitores."""
    gx, gy, gw, _ = cfg.GAME_REGION
    tela = sct.monitors[0]  # todos os monitores juntos
    x0 = max(tela["left"], gx - cfg.px(MARGEM_X))
    y0 = max(tela["top"], gy - cfg.px(MARGEM_CIMA))
    x1 = min(tela["left"] + tela["width"], gx + gw + cfg.px(MARGEM_X))
    y1 = min(tela["top"] + tela["height"], gy + cfg.px(MARGEM_BAIXO))
    return {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}


def _grab(sct, mon: dict) -> np.ndarray:
    return np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]


def captura_unica(sct, pasta: Path, n: int, regioes: dict) -> None:
    mon = _regiao_janela(sct)
    nome = f"tela_{n:03d}.png"
    cv2.imwrite(str(pasta / nome), _grab(sct, mon))
    regioes[nome] = mon
    print(f"  [F8] {nome} ({mon['width']}x{mon['height']})")


def rajada(sct, pasta: Path, n: int, segundos: float, regioes: dict) -> None:
    gx, gy, gw, gh = cfg.GAME_REGION
    mon = {"left": gx, "top": gy, "width": gw, "height": gh}
    print(f"  [F9] gravando {segundos:.0f}s... dispare agora")
    quadros, tempos = [], []
    t0 = time.perf_counter()
    while (t := time.perf_counter() - t0) < segundos:
        quadros.append(_grab(sct, mon))
        tempos.append(t)
    sub = pasta / f"rajada_{n:03d}"
    sub.mkdir()
    for i, q in enumerate(quadros):
        # compressão 1: PNG continua sem perda, só grava mais rápido
        cv2.imwrite(str(sub / f"f_{i:04d}.png"), q, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    with open(sub / "tempos.csv", "w", encoding="utf-8") as f:
        f.write("quadro,segundos\n")
        f.writelines(f"{i},{t:.5f}\n" for i, t in enumerate(tempos))
    regioes[sub.name] = mon
    fps = len(quadros) / max(tempos[-1], 1e-6) if tempos else 0
    print(f"  [F9] {sub.name}: {len(quadros)} quadros ({fps:.0f} fps)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segundos", type=float, default=4.0, help="duração da rajada do F9 (padrão 4)")
    args = ap.parse_args()

    pasta = PASTA_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta.mkdir(parents=True)
    regioes = {"GAME_REGION": dict(zip(("left", "top", "width", "height"), cfg.GAME_REGION))}
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
                    captura_unica(sct, pasta, n_tela, regioes)
                if _apertou(VK_F9, estado):
                    n_rajada += 1
                    rajada(sct, pasta, n_rajada, args.segundos, regioes)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
    with open(pasta / "regioes.json", "w", encoding="utf-8") as f:
        json.dump(regioes, f, indent=2)
    print(f"\n{n_tela} capturas e {n_rajada} rajadas em {pasta}")


if __name__ == "__main__":
    main()
