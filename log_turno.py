"""
log_turno.py — Diagnóstico de cada turno: o que o bot viu, previu e o que aconteceu.

Cada sessão do main.py cria logs/bot_N/ (N = maior número já usado + 1) com:

  sessao.log              tudo que passa por log(): também vai pro terminal
  turno_NNN_previsto.png  o scan + a trajetória prevista + o dano previsto por bloco
  turno_NNN_real.png      o rastro REAL das bolas (ciano) sobre a trajetória prevista
                          (laranja) nos primeiros segundos de voo

No texto, cada turno mostra:
  - o tabuleiro detectado como grade (HP por célula, triângulos, power-ups, bola);
  - a decisão: ângulo, de onde vem o score, top 5, vantagem sobre o 2o;
  - a previsão: dano em cada bloco e quais morrem;
  - o rastro: quanto do voo real bateu com o previsto e onde se afastou;
  - no turno seguinte, a CONFERÊNCIA: tabuleiro previsto (descido uma fileira)
    x tabuleiro lido, com uma pista do motivo de cada diferença (física, visão
    ou OCR do HP).

Só ASCII no texto: símbolos como ◢ quebram o print quando a saída é
redirecionada para arquivo no Windows (cp1252). O putText do OpenCV também só
desenha ASCII.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

import config as cfg
from vision import detect_retrieve_button

PASTA = Path(__file__).resolve().parent / "logs"

RASTRO_S = 2.5  # segundos de voo capturados depois do disparo (teto: para antes se a rodada acabar)
RETRIEVE_CHECK_S = 0.15  # intervalo entre checagens do botão retrieve durante a captura
# Distância (px de referência) até o caminho previsto pra um pixel do rastro
# contar como "no previsto". A bola real tem raio ~9px; a folga cobre o erro de
# posição da detecção e o arredondamento do path (1 ponto a cada 4 steps).
RASTRO_TOLERANCIA = cfg.px(14)

_pasta_sessao: Optional[Path] = None
_arquivo = None  # handle do sessao.log; None = só terminal

# Canto onde fica o CORPO do triângulo (TriangleOrientation), em 2 letras
_TRI = {
    "top_left": "tl",
    "top_right": "tr",
    "bottom_left": "bl",
    "bottom_right": "br",
    "unknown": "t?",
}
_PU = {
    "extra_ball": "+",
    "bonus_b": "B",
    "laser_h": "LH",
    "laser_v": "LV",
    "random": "R",
    "unknown": "?",
}
_LARG = 6  # largura de cada célula da grade

# Cores (BGR)
_LARANJA = (0, 200, 255)
_CIANO = (255, 255, 0)
_VERMELHO = (0, 0, 255)
_BRANCO = (255, 255, 255)
_AZUL = (255, 160, 0)


def _proxima_pasta() -> Path:
    """
    logs/bot_N/ com N = maior número já usado + 1 (bot_1 na primeira sessão).
    Só conta pastas bot_<número>: nomes antigos como bot_20260929_173907 ficam
    de fora da conta.
    """
    usados = [
        int(p.name[4:])
        for p in PASTA.glob("bot_*")
        if p.is_dir() and p.name[4:].isdigit()
    ]
    return PASTA / f"bot_{max(usados, default=0) + 1}"


def abrir() -> Path:
    """Cria logs/bot_N/ (ver _proxima_pasta) e abre o sessao.log. Devolve a pasta."""
    global _arquivo, _pasta_sessao
    # antigo: PASTA / f"bot_{datetime.now():%Y%m%d_%H%M%S}"
    PASTA.mkdir(exist_ok=True)
    _pasta_sessao = _proxima_pasta()
    _pasta_sessao.mkdir()
    _arquivo = open(_pasta_sessao / "sessao.log", "a", encoding="utf-8")
    _arquivo.write(f"sessao iniciada em {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return _pasta_sessao


def fechar():
    global _arquivo
    if _arquivo is not None:
        _arquivo.close()
        _arquivo = None


def log(msg: str = ""):
    """print() que também grava no sessao.log."""
    print(msg)
    if _arquivo is not None:
        _arquivo.write(msg + "\n")
        _arquivo.flush()


def cabecalho_turno(turno: int) -> str:
    return f"\n===== TURNO {turno} | {datetime.now():%H:%M:%S} " + "=" * 40


def _nome(r: int, c: int) -> str:
    """Célula no formato das janelas de debug: L1 = topo, L7 = fileira r0."""
    return f"L{7 - r}C{c}"


# ─────────────────────────────────────────────
# 1. O QUE O BOT VIU
# ─────────────────────────────────────────────


def _coluna(x: float) -> int:
    """Coluna (0-6) mais próxima de um X do frame."""
    centros = [(l + r) / 2 for l, r in zip(cfg.COL_LEFTS, cfg.COL_RIGHTS)]
    return min(range(len(centros)), key=lambda i: abs(centros[i] - x))


def _fileira(y: float) -> int:
    """Fileira (0 = baixo, 6 = topo) mais próxima de um Y do frame."""
    centros = [(t + b) / 2 for t, b in zip(cfg.LEVEL_TOPS, cfg.LEVEL_BOTS)]
    return min(range(len(centros)), key=lambda i: abs(centros[i] - y))


def tabuleiro(state) -> str:
    """
    Grade 7x7 do que a visão detectou, de cima (L1) para baixo (L7):

           C0    C1    C2 ...
      L1   .     30   12tl ...     <- 30 = quadrado com HP 30; 12tl = triângulo
      ...                             com o corpo no canto de cima-esquerda
      L7   .     .     .     <- r0: fileira do game over
                  ^ bola
    """
    celulas = {}
    for b in state.blocks:
        txt = str(b.hp)
        if b.block_type.value == "triangle":
            txt += _TRI.get(b.tri_orientation.value, "t?")
        if b.w > cfg.BLOCK_SIZE * 1.5:  # bloco grande 2x2
            txt += "#"
        if getattr(b, "hp_estimado", False):  # HP ilegível/corrigido (vision.ajustar_hp)
            txt += "?"
        chave = (b.row_index, b.col_index)
        celulas[chave] = txt if chave not in celulas else celulas[chave] + "|" + txt
    for p in state.powerups:
        chave = (_fileira(p.cy), _coluna(p.cx))
        celulas.setdefault(chave, _PU.get(p.power_up_type.value, "?"))

    linhas = ["       " + "".join(f"C{c}".center(_LARG) for c in range(7))]
    for r in range(6, -1, -1):
        cel = "".join(celulas.get((r, c), ".").center(_LARG) for c in range(7))
        fim = "   <- r0: game over se descer" if r == 0 else ""
        linhas.append(f"  L{7 - r} r{r}{cel}{fim}")
    if state.ball_x is not None:
        col = _coluna(state.ball_x)
        linhas.append(" " * (7 + col * _LARG + _LARG // 2) + "^ bola")
    linhas.append(
        "  legenda: HP (? = estimado) | tl/tr/bl/br triangulo (canto do corpo) | # bloco 2x2 | "
        "+ bola extra | B moeda | LH/LV laser | R direcao aleatoria"
    )
    return "\n".join(linhas)


def estado(state) -> str:
    """Bola, contagem e totais do scan, seguidos da grade."""
    if state.game_over:
        return f"  tela de game over ({state.white_ratio:.0%} branca)"
    n_tri = sum(1 for b in state.blocks if b.block_type.value == "triangle")
    hp_total = sum(b.hp for b in state.blocks)
    if state.ball_x is None:
        bola = "bola NAO encontrada"
    else:
        bola = (
            f"bola x={state.ball_x:.0f} y={state.ball_y:.0f} "
            f"(coluna C{_coluna(state.ball_x)})"
        )
    cab = (
        f"  {bola} | bolas (OCR): {state.detected_ball_count}\n"
        f"  {len(state.blocks)} blocos ({n_tri} triangulos), HP total {hp_total} | "
        f"{len(state.powerups)} power-ups"
    )
    return cab + "\n" + tabuleiro(state)


# ─────────────────────────────────────────────
# 2. O QUE O BOT PREVIU
# ─────────────────────────────────────────────


def _explica_score(t) -> str:
    """Score = soma dos termos com peso; mostra só os que contribuem."""
    partes = [
        (cfg.W_HIT * t.weighted_hits, f"hits {t.weighted_hits:.0f}x{cfg.W_HIT:g}"),
        (
            cfg.W_DESTROY * t.weighted_destroys,
            f"destruicoes {t.weighted_destroys:.0f}x{cfg.W_DESTROY:g}",
        ),
        (cfg.W_EXTRA_BALL * t.extra_balls, f"bola extra {t.extra_balls}x{cfg.W_EXTRA_BALL:g}"),
        (cfg.W_BONUS_B * t.bonus_b, f"moeda {t.bonus_b}x{cfg.W_BONUS_B:g}"),
        (cfg.W_LOOP * (t.total_hits // 10), f"loop {t.total_hits // 10}x{cfg.W_LOOP:g}"),
        (
            -cfg.W_ROW0_SURVIVE * t.row0_surviving_hp,
            f"HP vivo r0 -{t.row0_surviving_hp}x{cfg.W_ROW0_SURVIVE:g}",
        ),
    ]
    usados = [txt for valor, txt in partes if valor]
    return " + ".join(usados) if usados else "0"


def decisao(best, all_results, ball_count: int, sim_time: float) -> str:
    """Melhor ângulo, composição do score, top 5 e dano previsto por bloco."""
    ranking = sorted(all_results, key=lambda r: (r.score, r.total_hits), reverse=True)
    segundo = ranking[1] if len(ranking) > 1 else None
    vantagem = (
        f" | vantagem sobre o 2o ({segundo.angle:.0f}): {best.score - segundo.score:+.0f}"
        if segundo
        else ""
    )
    empatados = sum(1 for r in all_results if r.score == best.score)

    linhas = [
        f"  simulacao: {len(all_results)} angulos x {ball_count} bolas em {sim_time:.2f}s",
        f"  ESCOLHIDO {best.angle:.0f} graus | score {best.score:.0f}{vantagem}"
        + (f" | {empatados} angulos empatados" if empatados > 1 else ""),
        f"    score = {_explica_score(best.terms)}",
        f"    total: {best.total_hits} hits ({best.terms.danger_hits} em r0-r1), "
        f"{best.blocks_destroyed} destruidos, {best.powerups_collected} power-ups "
        f"(+{best.extra_balls_collected} bolas)",
        "  top 5: " + " | ".join(f"{r.angle:.0f}={r.score:.0f}" for r in ranking[:5]),
    ]

    atingidos = [h for h in best.hp_final if h[3] != h[2]]
    if atingidos:
        linhas.append("  dano previsto (celula hp_antes->hp_depois):")
        atingidos.sort(key=lambda h: (-h[0], h[1]))
        txt = [
            f"{_nome(r, c)} {ini}->{fim}" + (" MORRE" if fim == 0 else "")
            for r, c, ini, fim in atingidos
        ]
        for i in range(0, len(txt), 5):
            linhas.append("    " + "  ".join(txt[i : i + 5]))
    else:
        linhas.append("  dano previsto: nenhum bloco atingido")
    return "\n".join(linhas)


def _desenha_deteccao(vis: np.ndarray, state) -> None:
    """
    Contorno fino de cada bloco e power-up detectado, com o HP LIDO no canto
    de cima ("=30"; "=30t" se a visão achou que é triângulo). O número grande
    do próprio jogo continua visível, então erro de OCR salta aos olhos.
    """
    for b in state.blocks:
        tri = b.block_type.value == "triangle"
        if tri and len(b.vertices) >= 3:
            pts = b.vertices.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis, [pts], True, _AZUL, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), _AZUL, 1)
        _texto(vis, f"={b.hp}" + ("t" if tri else ""), (b.x + 2, b.y + 12), _CIANO, 0.35)
    for p in state.powerups:
        cv2.circle(vis, (int(p.cx), int(p.cy)), int(p.radius), _CIANO, 1)


def _texto(img, texto, org, cor, escala=0.45):
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1)


MAX_CAMINHOS = 8  # com dezenas de bolas, mais que isso vira borrão na imagem


def _caminhos_unicos(best, limite: Optional[int] = MAX_CAMINHOS):
    """
    Os paths distintos das bolas, na ordem de lançamento, até MAX_CAMINHOS
    (bolas seguem o mesmo caminho até algum bloco quebrar).
    """
    vistos, unicos = set(), []
    for p in best.ball_paths or [best.path]:
        if len(p) < 2:
            continue
        chave = (len(p), p[0], p[len(p) // 2], p[-1])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(p)
            if limite is not None and len(unicos) >= limite:
                break
    return unicos


def _salvar(nome: str, img: np.ndarray) -> Optional[str]:
    if _pasta_sessao is None:
        return None
    caminho = _pasta_sessao / nome
    return nome if cv2.imwrite(str(caminho), img) else None


def salvar_previsto(turno: int, state, best) -> None:
    """
    turno_NNN_previsto.png: scan + trajetória da 1a bola (laranja grosso) +
    trajetórias das bolas seguintes que mudaram após blocos quebrarem (laranja
    fino) + colisões da 1a bola (vermelho) + dano previsto em cada bloco.
    """
    vis = state.raw_frame.copy()
    _desenha_deteccao(vis, state)
    bx, by = state.ball_x, state.ball_y

    for p in _caminhos_unicos(best)[1:]:
        pts = np.array([(bx, by)] + p, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, _LARANJA, 1, cv2.LINE_AA)
    if len(best.path) >= 2:
        pts = np.array([(bx, by)] + best.path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, _LARANJA, 2, cv2.LINE_AA)
    for cx, cy in best.collisions:
        cv2.circle(vis, (int(cx), int(cy)), 3, _VERMELHO, -1)

    # Dano previsto em cima de cada bloco atingido
    por_celula = {(r, c): (ini, fim) for r, c, ini, fim in best.hp_final}
    for b in state.blocks:
        ini, fim = por_celula.get((b.row_index, b.col_index), (b.hp, b.hp))
        if fim != ini:
            txt = "X" if fim == 0 else f"->{fim}"
            _texto(vis, txt, (b.x + 3, b.y + b.h - 5), _VERMELHO if fim == 0 else _BRANCO, 0.4)

    _texto(vis, f"T{turno} PREVISTO {best.angle:.0f}deg score {best.score:.0f}", (6, 18), _BRANCO)
    _texto(
        vis,
        f"{best.total_hits} hits  {best.blocks_destroyed} destruidos  +{best.extra_balls_collected} bolas",
        (6, 36),
        _LARANJA,
        0.4,
    )
    nome = _salvar(f"turno_{turno:03d}_previsto.png", vis)
    if nome:
        log(f"  imagem: {nome}")


# ─────────────────────────────────────────────
# 3. O QUE ACONTECEU DE VERDADE
# ─────────────────────────────────────────────


def capturar_rastro(base_frame: np.ndarray, segundos: float = RASTRO_S) -> dict:
    """
    Grava o GAME_REGION o mais rápido possível durante `segundos` depois do
    disparo e acumula onde apareceram pixels de bola (BALL_HSV) que NÃO estavam
    no frame de antes do tiro — o que sobra é o caminho real das bolas.

    Também vigia o botão retrieve (a cada RETRIEVE_CHECK_S): se ele aparece e
    some, a rodada acabou e a captura para ali. Antes a captura sempre durava
    RASTRO_S e, com o acelerar ligado, a rodada terminava dentro dela — o main
    então esperava 5s por um botão que já tinha sumido (todo turno do bot_1).

    Devolve {"primeiro": mapa com o índice do 1o frame em que cada pixel acendeu
    (-1 = nunca), "tempos": instante de cada frame, "frames": quantidade,
    "segundos": duração real, "retrieve_visto", "rodada_acabou": bool}.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    base = cv2.inRange(cv2.cvtColor(base_frame, cv2.COLOR_BGR2HSV), cfg.BALL_HSV_LOW, cfg.BALL_HSV_HIGH)
    base = cv2.dilate(base, kernel)  # tolera 2px de tremida no que já era branco

    rx, ry, rw, rh = cfg.GAME_REGION
    mon = {"left": rx, "top": ry, "width": rw, "height": rh}
    primeiro = np.full((rh, rw), -1, dtype=np.int32)
    tempos = []
    visto = acabou = False
    proxima_checagem = 0.0
    t0 = time.perf_counter()
    with mss.mss() as sct:
        while time.perf_counter() - t0 < segundos:
            frame = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bola = cv2.inRange(hsv, cfg.BALL_HSV_LOW, cfg.BALL_HSV_HIGH)
            bola[base > 0] = 0
            novos = (bola > 0) & (primeiro < 0)
            primeiro[novos] = len(tempos)
            tempos.append(time.perf_counter() - t0)
            if tempos[-1] >= proxima_checagem:
                proxima_checagem = tempos[-1] + RETRIEVE_CHECK_S
                if detect_retrieve_button():
                    visto = True
                elif visto:
                    acabou = True
                    break
    return {
        "primeiro": primeiro,
        "tempos": tempos,
        "frames": len(tempos),
        "segundos": tempos[-1] if tempos else 0.0,
        "retrieve_visto": visto,
        "rodada_acabou": acabou,
    }


# Janela (px de referência a partir da bola) usada pra medir o ângulo de saída:
# longe o bastante pra sair de cima da bola/robô, perto o bastante pra ainda ser
# o 1o segmento do voo (antes da 1a batida).
ANGULO_RAIO_MIN = cfg.px(60)
ANGULO_RAIO_MAX = cfg.px(220)
ANGULO_JANELA = 12  # graus em volta do escolhido que contam como "1o segmento"
ANGULO_MIN_PIXELS = 30


def angulo_saida(rastro_mask: np.ndarray, bx: float, by: float, escolhido: float):
    """
    Ângulo real (graus, 0 = direita, 90 = cima) do 1o segmento do voo: mediana
    do ângulo, visto da bola, dos pixels do rastro entre ANGULO_RAIO_MIN e
    ANGULO_RAIO_MAX dela e a menos de ANGULO_JANELA graus do escolhido.

    Medido nos 46 turnos do logs/bot_1: desvio de 2,3 graus. Ajustar uma reta
    (cv2.fitLine, L2 ou Huber) nos mesmos pixels deu desvio de ~31 graus — o
    trecho é uma faixa curta e grossa de bolas, e a direção fica mal definida.
    Limitação: se a origem real do disparo não for a bola detectada, a medida
    carrega esse erro. Devolve (ângulo, n_pixels) ou None.
    """
    ys, xs = np.nonzero(rastro_mask)
    if len(xs) == 0:
        return None
    dx, dy = xs - bx, by - ys
    d = np.hypot(dx, dy)
    a = np.degrees(np.arctan2(dy, dx))
    sel = (d > ANGULO_RAIO_MIN) & (d < ANGULO_RAIO_MAX) & (dy > 5) & (np.abs(a - escolhido) < ANGULO_JANELA)
    if np.count_nonzero(sel) < ANGULO_MIN_PIXELS:
        return None
    return float(np.median(a[sel])), int(np.count_nonzero(sel))


def comparar_rastro(state, best, rastro: dict) -> str:
    """
    Mede o quanto o voo real bateu com o previsto:
      - o ângulo real de saída contra o escolhido (erro de execução do tiro);
      - % dos pixels do rastro a menos de RASTRO_TOLERANCIA da trajetória da
        1a bola (antigo: de qualquer um dos caminhos desenhados, o que deixava
        a conta generosa demais — o turno 13 do bot_1 deu 94% e errou tudo);
      - o primeiro instante em que um grupo de pixels apareceu LONGE da 1a bola
        (onde a bola real se separou da simulada) e a célula mais perto dali.
    """
    primeiro = rastro["primeiro"]
    aceso = primeiro >= 0
    n_aceso = int(np.count_nonzero(aceso))
    if rastro["frames"] == 0 or n_aceso == 0:
        return f"  rastro real: nada capturado ({rastro['frames']} frames em {rastro['segundos']:.1f}s)"

    # antigo: _mascara_previsto(best, ...) com todos os caminhos únicos
    prev = np.zeros(primeiro.shape, dtype=np.uint8)
    pts = np.array([(state.ball_x, state.ball_y)] + best.path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(prev, [pts], False, 255, 1)
    dist = cv2.distanceTransform(cv2.bitwise_not(prev), cv2.DIST_L2, 3)
    fora = aceso & (dist > RASTRO_TOLERANCIA)
    pct = 100.0 * (n_aceso - np.count_nonzero(fora)) / n_aceso
    fps = rastro["frames"] / max(rastro["tempos"][-1], 1e-6)

    medido = angulo_saida(aceso, state.ball_x, state.ball_y, best.angle)
    if medido:
        real, n = medido
        saida = f"angulo real de saida {real:.1f} (escolhido {best.angle:.0f}, erro {real - best.angle:+.1f}, {n}px)"
    else:
        saida = "angulo real de saida: sem pixels suficientes"

    linhas = [
        f"  rastro real: {rastro['frames']} frames ({fps:.0f} fps) em {rastro['segundos']:.1f}s | "
        f"{pct:.0f}% do rastro a <= {RASTRO_TOLERANCIA}px da 1a bola",
        f"    {saida}",
    ]
    # 1o frame com um grupo relevante de pixels fora do previsto
    MIN_PIXELS = 12
    for k in range(rastro["frames"]):
        ys, xs = np.nonzero(fora & (primeiro == k))
        if len(xs) >= MIN_PIXELS:
            x, y = float(xs.mean()), float(ys.mean())
            linhas.append(
                f"    primeira divergencia aos {rastro['tempos'][k]:.2f}s em "
                f"x={x:.0f} y={y:.0f} (perto de {_nome(_fileira(y), _coluna(x))})"
            )
            break
    else:
        linhas.append("    sem divergencia relevante: o voo real seguiu o previsto")
    return "\n".join(linhas)


def salvar_contador(turno: int, lido: int, usado: Optional[int] = None) -> None:
    """
    turno_NNN_contador.png: BALL_COUNT_REGION ampliada 3x, com o número que o
    OCR leu (e o usado, se o filtro de coerência corrigiu) escrito embaixo. Sem
    isso não dá pra saber, depois, se um "x43" lido como 1 foi erro do OCR ou o
    contador mostrando outra coisa. Com o valor usado, as imagens viram um
    conjunto rotulado pra refazer os templates dos dígitos no tamanho nativo.
    """
    bx, by, bw, bh = cfg.BALL_COUNT_REGION
    with mss.mss() as sct:
        img = np.array(sct.grab({"left": bx, "top": by, "width": bw, "height": bh}), dtype=np.uint8)[:, :, :3]
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rotulo = f"T{turno} OCR leu {lido}" + (f", usado {usado}" if usado is not None and usado != lido else "")
    _texto(img, rotulo, (6, img.shape[0] - 7), _BRANCO)
    nome = _salvar(f"turno_{turno:03d}_contador.png", img)
    if nome:
        log(f"  imagem: {nome}")


def salvar_real(turno: int, state, best, rastro: dict) -> None:
    """
    turno_NNN_real.png: frame de antes do tiro, escurecido, com o previsto em
    laranja e o rastro real por cima. O rastro vai de ciano (início do voo) a
    magenta (fim), pra dar a ordem no tempo.
    """
    log(comparar_rastro(state, best, rastro))

    vis = (state.raw_frame * 0.45).astype(np.uint8)
    _desenha_deteccao(vis, state)
    bx, by = state.ball_x, state.ball_y
    for p in _caminhos_unicos(best):
        pts = np.array([(bx, by)] + p, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, _LARANJA, 1, cv2.LINE_AA)

    primeiro = rastro["primeiro"]
    n = max(rastro["frames"] - 1, 1)
    ys, xs = np.nonzero(primeiro >= 0)
    frac = primeiro[ys, xs].astype(np.float32) / n
    # ciano (255,255,0) -> magenta (255,0,255) em BGR
    vis[ys, xs] = np.stack(
        [np.full_like(frac, 255), (255 * (1 - frac)), (255 * frac)], axis=1
    ).astype(np.uint8)

    _texto(vis, f"T{turno} REAL (ciano->magenta) x PREVISTO (laranja)", (6, 18), _BRANCO)
    _texto(vis, f"{best.angle:.0f}deg  {rastro['frames']} frames em {rastro['segundos']:.1f}s", (6, 36), _CIANO, 0.4)
    nome = _salvar(f"turno_{turno:03d}_real.png", vis)
    if nome:
        log(f"  imagem: {nome}")


# ─────────────────────────────────────────────
# 4 e 5. DIFERENÇAS E PISTAS DO MOTIVO
# ─────────────────────────────────────────────


def conferencia(best_anterior, bolas_anterior: int, state) -> Optional[str]:
    """
    Compara o tabuleiro que a simulação previu com o que o scan novo leu.

    Depois de cada rodada tudo desce uma fileira: um bloco que a simulação
    deixou em (r, c) com HP h deveria aparecer agora em (r-1, c) com HP h.
    A fileira do topo (r6) é nova, então fica fora da conta.

    Cada diferença vem com uma pista:
      - HP maior que o lido antes: erro de OCR (HP nunca sobe);
      - bloco que a simulação não previa atingir mudou: o voo real foi outro;
      - bloco onde antes não havia nada: erro de visão (bloco fantasma/perdido);
      - dano diferente do previsto: a física errou a quantidade de hits.
    """
    if best_anterior is None or not best_anterior.hp_final:
        return None

    antes = {}  # (r, c) no tabuleiro NOVO -> (hp lido antes, hp previsto)
    for r, c, ini, fim in best_anterior.hp_final:
        if r >= 1:
            antes[(r - 1, c)] = (ini, fim)
    esperado = {cel: fim for cel, (_ini, fim) in antes.items() if fim > 0}
    lido = {(b.row_index, b.col_index): b.hp for b in state.blocks if b.row_index < 6}

    def pista(ini, fim, agora):
        if agora > ini:
            return "HP subiu: erro de OCR (agora ou no turno anterior)"
        if fim == ini:
            return "simulacao nao previa atingir: o voo real foi outro"
        return "dano real diferente do previsto: fisica errou os hits"

    certos, diferencas = 0, []
    for cel, hp in sorted(esperado.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        ini, fim = antes[cel]
        nome = _nome(*cel)
        if cel not in lido:
            motivo = (
                "simulacao nao previa atingir: o voo real foi outro"
                if fim == ini
                else "morreu antes do previsto: fisica subestimou os hits"
            )
            diferencas.append(f"{nome}: antes {ini}, previa {fim}, SUMIU -> {motivo}")
        elif lido[cel] == hp:
            certos += 1
        else:
            agora = lido[cel]
            diferencas.append(
                f"{nome}: antes {ini}, previa {fim} ({fim - ini:+d}), "
                f"leu {agora} ({agora - ini:+d}) -> {pista(ini, fim, agora)}"
            )
    for cel, hp in sorted(lido.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        if cel in esperado:
            continue
        nome = _nome(*cel)
        if cel in antes:
            ini, fim = antes[cel]
            # antigo: sempre "fisica superestimou os hits (ou OCR leu HP errado antes)"
            motivo = (
                "HP subiu: erro de OCR (agora ou no turno anterior)"
                if hp > ini
                else "fisica superestimou os hits (ou OCR leu HP errado antes)"
            )
            diferencas.append(f"{nome}: antes {ini}, previa destruir, leu {hp} -> {motivo}")
        else:
            diferencas.append(
                f"{nome}: leu {hp} onde antes nao havia bloco -> erro de visao "
                "(bloco perdido no turno anterior ou fantasma agora)"
            )

    bolas_prev = bolas_anterior + best_anterior.extra_balls_collected
    linhas = [
        f"  CONFERENCIA do turno anterior ({best_anterior.angle:.0f} graus): "
        f"{certos}/{len(esperado)} blocos vivos com o HP previsto, "
        f"{len(diferencas)} diferencas"
    ]
    linhas += [f"    {d}" for d in diferencas]
    linhas.append(
        f"    bolas: previa {bolas_prev} ({bolas_anterior} + "
        f"{best_anterior.extra_balls_collected} extra), leu {state.detected_ball_count}"
        + ("" if bolas_prev == state.detected_ball_count else "  <- DIFERENTE")
    )
    return "\n".join(linhas)
