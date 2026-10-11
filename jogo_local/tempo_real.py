"""
tempo_real.py — Roda o Jogo no relógio de verdade, ligado ao BackendLocal:
o que o bot faz com o mouse vira ação no jogo e cada tick vira imagem nova.

    import tela
    from jogo_local.backend import BackendLocal
    from jogo_local.jogo import Jogo
    from jogo_local.tempo_real import TempoReal

    backend = BackendLocal()
    TempoReal(Jogo(semente=1), backend).iniciar()
    tela.usar(backend)   # daqui em diante o bot joga o jogo local

Mouse (coordenadas de TELA, as mesmas do jogo real):
  - arrastar começando perto da bola parada, pra cima = tiro na direção do
    arrasto, SEM viés de mira; a 1a bola sai ATRASO_DISPARO_S depois de
    soltar (medido no jogo real);
  - clicar no acelerar (com a rodada rolando) = 240 ticks/s depois de
    ATRASO_ACELERAR_S; clicar no Retrieve = recolhe as bolas;
  - clicar com a tela de fim = partida nova.
Ritmo: 50 ticks/s (240 acelerado), como o jogo real. Com um
relogio.RelogioVirtual, não há thread: o jogo anda quando o relógio virtual
anda (os sleeps do bot), e uma partida roda sem esperar o tempo real.
"""

import math
import threading
import time
from typing import List, Optional, Tuple

import config as cfg
from jogo_local import fisica as fis
from jogo_local import geometria as geo
from jogo_local.backend import BackendLocal
from jogo_local.jogo import Jogo

RAIO_PEGAR_BOLA = 60  # px: o arrasto tem que começar perto da bola parada
RAIO_BOTAO_ACELERAR = 30


class TempoReal:
    def __init__(self, jogo: Jogo, backend: BackendLocal, relogio=None):
        self.jogo = jogo
        self.backend = backend
        self._trava = threading.Lock()
        self._agenda: List[Tuple[float, str, Optional[float]]] = []  # (quando, ação, ângulo)
        self._acelerado = False
        self._parar = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.tiros: List[Tuple[float, float]] = []  # (instante, ângulo) — pra conferir
        self.relogio = relogio
        self._proximo = relogio.agora() if relogio is not None else time.perf_counter()  # próximo tick
        backend.ouvinte = self._mouse
        backend.atualizar(jogo.tela())
        if relogio is not None:
            relogio.ouvintes.append(self._ate)

    # ── controle ────────────────────────────────────────────────────────────
    def iniciar(self) -> "TempoReal":
        if self.relogio is None:  # relógio virtual: quem anda é o relógio
            self._thread = threading.Thread(target=self._rodar, daemon=True)
            self._thread.start()
        return self

    def parar(self) -> None:
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ── mouse do bot ────────────────────────────────────────────────────────
    def _mouse(self, ev: tuple) -> None:
        agora, tipo = ev[0], ev[1]
        gx, gy = cfg.GAME_REGION[:2]
        with self._trava:
            if tipo == "arrastar":
                _, _, x, y, dx, dy, dur = ev
                bx, by = gx + self.jogo.est.bola_x, gy + geo.BOLA_Y
                if (self.jogo.rodada is None and not self.jogo.game_over and dy < 0
                        and math.hypot(x - bx, y - by) <= RAIO_PEGAR_BOLA * cfg.ESCALA):
                    ang = math.degrees(math.atan2(-dy, dx))
                    # ev[0] é o instante em que o mouse foi solto
                    self._agenda.append((agora + fis.ATRASO_DISPARO_S, "disparo", ang))
            elif tipo == "clicar":
                _, _, x, y = ev
                if self.jogo.game_over:
                    self._agenda.append((agora, "novo_jogo", None))
                elif self.jogo.rodada is not None or any(a == "disparo" for _, a, _ in self._agenda):
                    rx, ry, rw, rh = cfg.RETRIEVE_BUTTON_REGION
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        self._agenda.append((agora, "retrieve", None))
                    elif math.hypot(x - cfg.SPEED_UP_X, y - cfg.SPEED_UP_Y) <= RAIO_BOTAO_ACELERAR * cfg.ESCALA:
                        self._agenda.append((agora + fis.ATRASO_ACELERAR_S, "acelerar", None))

    # ── relógio ─────────────────────────────────────────────────────────────
    def _rodar(self) -> None:
        while not self._parar.is_set():
            self._ate(time.perf_counter())
            time.sleep(0.002)

    def _ate(self, agora: float) -> None:
        """Leva o jogo até o instante `agora`: ticks e ações agendadas na
        ordem certa (um acelerar no meio do intervalo vale dali em diante)."""
        mudou = False
        with self._trava:
            while True:
                pendentes = sorted(a for a in self._agenda if a[0] <= agora)
                limite = pendentes[0][0] if pendentes else agora
                mudou |= self._ticks_ate(limite)
                if not pendentes:
                    break
                quando, acao, ang = pendentes[0]
                self._agenda.remove(pendentes[0])
                mudou |= self._executar(acao, ang, quando)
            if mudou:
                self.backend.atualizar(self.jogo.tela())

    def _ticks_ate(self, t: float) -> bool:
        if self.jogo.rodada is None:
            self._proximo = t  # parado: o 1o tick da próxima rodada sai na hora do disparo
            return False
        periodo = 1.0 / (fis.TICKS_POR_S_ACELERADO if self._acelerado else fis.TICK_HZ)
        andou = False
        while self._proximo <= t and self.jogo.rodada is not None:
            self.jogo.passo()
            self._proximo += periodo
            andou = True
        if self.jogo.rodada is None:
            self._acelerado = False
            self._proximo = t
        return andou

    def _executar(self, acao: str, ang: Optional[float], agora: float) -> bool:
        if acao == "disparo":
            if self.jogo.disparar(ang):
                self.tiros.append((agora, ang))
                return True
        elif acao == "acelerar" and self.jogo.rodada is not None:
            self._acelerado = True
        elif acao == "retrieve":
            self.jogo.retrieve()
            return True
        elif acao == "novo_jogo":
            self.jogo.novo_jogo()
            return True
        return False
