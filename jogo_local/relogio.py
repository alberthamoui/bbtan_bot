"""
relogio.py — Relógio virtual pra rodar o bot no jogo local sem esperar o
tempo real.

O bot espera o jogo de três jeitos: time.sleep (entre checagens, antes do
acelerar...), laços que olham time.monotonic/perf_counter e capturas de tela
em sequência (o rastro). Com o relógio virtual:
  - sleep(s) não dorme: avança o relógio s segundos, e o jogo local anda
    junto (os mesmos ticks que andaria no tempo real);
  - monotonic/perf_counter/time leem o relógio virtual;
  - cada captura de tela custa CUSTO_CAPTURA_S (um quadro de 60 fps, como o
    mss no jogo real), senão um laço "capture até passar X s" nunca acaba.
O que continua levando tempo de verdade é só o processamento do bot (a
simulação dos ângulos) — e isso o jogo real também espera parado.

    relogio = RelogioVirtual()
    relogio.instalar(vision, executor, log_turno, jogo_local.backend)
    relogio.instalar(main, perf_counter=False)  # mede a simulação em tempo real

Determinístico: mesma seed + mesmo bot = mesma partida.
"""

import time as _time
from typing import Callable, List

CUSTO_CAPTURA_S = 1.0 / 60


class _TempoVirtual:
    """Faz as vezes do módulo `time` dentro de um módulo do bot."""

    def __init__(self, relogio: "RelogioVirtual", perf_counter: bool):
        self._r = relogio
        self._perf = perf_counter

    def sleep(self, s: float) -> None:
        self._r.avancar(s)

    def monotonic(self) -> float:
        return self._r.agora()

    def perf_counter(self) -> float:
        return self._r.agora() if self._perf else _time.perf_counter()

    def time(self) -> float:
        return self._r.inicio_epoca + self._r.agora()

    def __getattr__(self, nome):  # strftime, localtime... continuam os de verdade
        return getattr(_time, nome)


class RelogioVirtual:
    def __init__(self):
        self.t = 0.0
        self.inicio_epoca = _time.time()
        self.ouvintes: List[Callable[[float], None]] = []  # chamados com o instante novo

    def agora(self) -> float:
        return self.t

    def avancar(self, dt: float) -> None:
        if dt <= 0:
            return
        self.t += dt
        for f in self.ouvintes:
            f(self.t)

    def instalar(self, *modulos, perf_counter: bool = True) -> None:
        """Troca o `time` desses módulos pelo relógio virtual. perf_counter=False
        deixa o perf_counter de verdade (ex: medir quanto a simulação demorou)."""
        for m in modulos:
            m.time = _TempoVirtual(self, perf_counter)
