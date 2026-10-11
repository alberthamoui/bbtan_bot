"""
tela.py — Único ponto por onde o bot VÊ a tela e MEXE o mouse.

Por padrão usa a tela e o mouse de verdade (mss + pyautogui), exatamente
como antes. O jogo local (jogo_local/) troca o "backend": no modo sem tela a
captura devolve a imagem que o jogo local desenhou na memória e o mouse vai
direto para ele, sem nada aparecer no monitor — o resto do bot (visão,
simulação, log) não muda.

Coordenadas sempre ABSOLUTAS de tela, como o resto do bot (cfg.GAME_REGION
etc.). capturar() devolve BGR uint8 (h, w, 3), como o OpenCV espera.
"""

import threading
from typing import Optional

import numpy as np


class BackendReal:
    """Tela e mouse de verdade (mss + pyautogui)."""

    def __init__(self):
        # mss não pode ser compartilhado entre threads (o Gravador do modo
        # --referencia captura numa thread própria): uma instância por thread
        self._local = threading.local()

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            import mss

            sct = self._local.sct = mss.mss()
        return sct

    def capturar(self, mon: dict) -> np.ndarray:
        return np.array(self._sct().grab(mon), dtype=np.uint8)[:, :, :3]

    def monitores(self) -> dict:
        """Retângulo que cobre todos os monitores (mss.monitors[0])."""
        return dict(self._sct().monitors[0])

    def mover(self, x: int, y: int, duracao: float = 0.0) -> None:
        import pyautogui

        pyautogui.moveTo(x, y, duration=duracao)

    def arrastar(self, dx: int, dy: int, duracao: float) -> None:
        """Arrasta a partir da posição atual do mouse (botão esquerdo)."""
        import pyautogui

        pyautogui.drag(dx, dy, duration=duracao, button="left")

    def clicar(self, x: int, y: int) -> None:
        import pyautogui

        pyautogui.click(x, y)


_backend = BackendReal()


def usar(backend) -> None:
    """Troca o backend (ex: jogo_local.BackendLocal). usar(None) volta pro real."""
    global _backend
    _backend = backend if backend is not None else BackendReal()


def backend():
    return _backend


def capturar(mon: dict) -> np.ndarray:
    """Captura o retângulo {left, top, width, height} da tela. BGR uint8."""
    return _backend.capturar(mon)


def capturar_regiao(regiao: tuple) -> np.ndarray:
    """Mesmo que capturar(), recebendo (x, y, w, h) como as regiões do config."""
    x, y, w, h = regiao
    return _backend.capturar({"left": x, "top": y, "width": w, "height": h})


def monitores() -> dict:
    return _backend.monitores()


def mover(x: int, y: int, duracao: float = 0.0) -> None:
    _backend.mover(int(x), int(y), duracao)


def arrastar(dx: int, dy: int, duracao: float) -> None:
    _backend.arrastar(int(dx), int(dy), duracao)


def clicar(x: int, y: int) -> None:
    _backend.clicar(int(x), int(y))


def nome_backend() -> Optional[str]:
    return type(_backend).__name__
