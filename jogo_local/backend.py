"""
backend.py — Liga o jogo local ao bot pelo tela.py, sem tela nenhuma.

    import tela
    from jogo_local.backend import BackendLocal
    backend = BackendLocal(estado)
    tela.usar(backend)           # daqui em diante o bot "vê" o jogo local
    ...
    backend.atualizar(novo_estado)

A "tela" é só a janela do jogo, posicionada exatamente onde o emulador fica
(o canto do tabuleiro em cfg.GAME_REGION), então todas as regiões absolutas do
config (contador, retrieve, fim de rodada...) caem no mesmo lugar. Fora da
janela, devolve a cor de fundo.

O mouse (mover/arrastar/clicar) é guardado em `eventos` com a posição na
TELA e repassado pro `ouvinte` (tempo_real.TempoReal), que vira ação no
jogo. Como o pyautogui, mover/arrastar demoram a `duracao` pedida (com
`esperar=False`, voltam na hora — pra rodar sem relógio).
"""

import threading
import time
from typing import Callable, List, Optional

import numpy as np

import config as cfg
from jogo_local.desenho import Desenhista
from jogo_local.estado import Estado


class BackendLocal:
    def __init__(self, estado: Optional[Estado] = None, desenhista: Optional[Desenhista] = None, esperar: bool = True):
        self.des = desenhista or Desenhista()
        self.ox = cfg.GAME_REGION[0] - self.des.bx  # canto da janela na tela
        self.oy = cfg.GAME_REGION[1] - self.des.by
        self.altura, self.largura = self.des.fundo.shape[:2]
        self._trava = threading.Lock()  # o bot captura de mais de uma thread
        self._estado = estado or Estado()
        self._imagem: Optional[np.ndarray] = None
        self.eventos: List[tuple] = []  # (instante, tipo, x, y[, dx, dy, duração])
        self.ouvinte: Optional[Callable[[tuple], None]] = None
        self.esperar = esperar
        self._mouse = (0, 0)

    # ── estado ──────────────────────────────────────────────────────────────
    def atualizar(self, estado: Estado) -> None:
        with self._trava:
            self._estado = estado
            self._imagem = None

    def imagem(self) -> np.ndarray:
        """A janela inteira desenhada (só redesenha quando o estado muda)."""
        with self._trava:
            if self._imagem is None:
                self._imagem = self.des.desenhar(self._estado)
            return self._imagem

    # ── interface do tela.py ────────────────────────────────────────────────
    def capturar(self, mon: dict) -> np.ndarray:
        img = self.imagem()
        x, y = mon["left"] - self.ox, mon["top"] - self.oy
        w, h = mon["width"], mon["height"]
        out = np.empty((h, w, 3), np.uint8)
        out[:] = self.des.cor_fundo.astype(np.uint8)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.largura, x + w), min(self.altura, y + h)
        if x1 > x0 and y1 > y0:
            out[y0 - y : y1 - y, x0 - x : x1 - x] = img[y0:y1, x0:x1]
        return out

    def monitores(self) -> dict:
        return {"left": self.ox, "top": self.oy, "width": self.largura, "height": self.altura}

    def _evento(self, ev: tuple) -> None:
        self.eventos.append(ev)
        if self.ouvinte is not None:
            self.ouvinte(ev)

    def mover(self, x: int, y: int, duracao: float = 0.0) -> None:
        if self.esperar and duracao > 0:
            time.sleep(duracao)
        self._mouse = (x, y)
        self._evento((time.perf_counter(), "mover", x, y))

    def arrastar(self, dx: int, dy: int, duracao: float) -> None:
        """O instante do evento é o do mouse SOLTO (fim do arrasto)."""
        x, y = self._mouse
        if self.esperar and duracao > 0:
            time.sleep(duracao)
        self._mouse = (x + dx, y + dy)
        self._evento((time.perf_counter(), "arrastar", x, y, dx, dy, duracao))

    def clicar(self, x: int, y: int) -> None:
        self._mouse = (x, y)
        self._evento((time.perf_counter(), "clicar", x, y))
