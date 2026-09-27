"""
main.py — Loop principal do BBTAN Bot.

Pipeline por turno:
  1. Captura tela e detecta blocos/bola (vision.py)
  2. Simula todos os ângulos e encontra o melhor (physics.py)
  3. Executa o disparo no emulador (executor.py)
  4. Espera o turno acabar
  5. Repete

Uso:
  python main.py              -> Roda o bot
  python main.py --debug      -> Mostra visualização a cada turno
  python main.py --dry-run    -> Simula sem executar ações no mouse
  python main.py --prints     -> Roda o bot salvando um print da GAME_REGION
                                 a cada turno em dirty-prints/
"""

import argparse
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2

import config as cfg
from vision import scan_game, detect_retrieve_button, GameState
from physics import find_best_angle
from executor import execute_shot, click_retrieve_button
from tools.visualizar_masks import (
    run_masks_window,
    update_level_display,
    update_shot_display,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BBTAN Bot")
    parser.add_argument(
        "--debug", action="store_true", help="Mostra visualização de debug a cada turno"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simula sem mover o mouse"
    )
    parser.add_argument(
        "--smart-wait",
        action="store_true",
        help="Usa detecção visual para esperar turno (mais lento)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay inicial em segundos antes de começar",
    )
    parser.add_argument(
        "--max-turns", type=int, default=0, help="Limite de turnos (0 = infinito)"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Visualizacao animada do dry-run (Phase 1: angulos candidatos, "
        "Phase 2: bola percorrendo a melhor trajetoria). Nao executa disparos.",
    )
    parser.add_argument(
        "--verbose-vision",
        action="store_true",
        help="Habilita prints de debug detalhados do modulo vision",
    )
    parser.add_argument(
        "--prints",
        action="store_true",
        help="Salva um print da tela do jogo (GAME_REGION) a cada turno, "
        "antes da jogada, na pasta dirty-prints/",
    )
    return parser.parse_args()


PRINTS_DIR = Path(__file__).resolve().parent / "dirty-prints"


def salvar_print(frame, turn: int) -> None:
    """
    Salva o frame CRU do scan (sem as anotações do debug_frame) em PRINTS_DIR.
    Nome com data/hora + turno, pra não sobrescrever prints de outras sessões:
    20260927_011745_turno012.png
    """
    PRINTS_DIR.mkdir(exist_ok=True)
    nome = f"{datetime.now():%Y%m%d_%H%M%S}_turno{turn:03d}.png"
    if cv2.imwrite(str(PRINTS_DIR / nome), frame):
        print(f"  [print] dirty-prints/{nome}")
    else:
        print(f"  [!] Falha ao salvar print dirty-prints/{nome}")


def make_progress_bar(prefix: str = "  Simulando", width: int = 28):
    """
    Devolve um callback (feitos, total) que desenha uma barra de progresso na
    mesma linha do terminal:

        Simulando [::::::::::::::____________] -  50%   72/144   ~2.0s

    Detalhes que importam:
      - Redesenha só quando o inteiro do percentual muda (senão seriam 144
        writes por turno, o que pisca feio).
      - Estima o tempo restante pela média corrida, e só a partir de 10% — o
        custo por ângulo varia ~100x (0.9ms a 90ms) conforme quantos blocos a
        bola encontra, então uma estimativa com poucas amostras chuta alto.
      - Ao chegar em 100% LIMPA a linha, porque logo em seguida run_turn imprime
        o resultado com o tempo total ("Simulacao: X.XXs").
      - Se a saída não for um terminal (pipe, redirecionamento), devolve None:
        aí find_best_angle nem chama o callback e o log não vira sopa de '\\r'.
    """
    if not sys.stdout.isatty():
        return None

    state = {"t0": None, "last_pct": -1}

    def report(done: int, total: int):
        now = time.perf_counter()
        if state["t0"] is None:
            state["t0"] = now

        pct = int(100 * done / total)
        if pct == state["last_pct"] and done < total:
            return  # nada de novo pra mostrar
        state["last_pct"] = pct

        if done >= total:
            # Limpa a linha inteira — o resumo do turno vem logo abaixo
            sys.stdout.write("\r" + " " * (width + 40) + "\r")
            sys.stdout.flush()
            return

        # ETA pela media corrida. O custo por angulo varia ~100x de forma
        # imprevisivel (depende do tabuleiro), entao a estimativa e grosseira no
        # comeco e vai se corrigindo; so mostra depois de 10% pra nao chutar alto.
        elapsed = now - state["t0"]
        restante = (elapsed / done) * (total - done) if done else 0.0
        eta = f"   ~{restante:.1f}s" if pct >= 10 else ""

        filled = int(width * done / total)
        bar = ":" * filled + "_" * (width - filled)
        sys.stdout.write(f"\r{prefix} [{bar}] - {pct:3d}%   {done}/{total}{eta}")
        sys.stdout.flush()

    return report


def aguardar_fim_da_rodada(dry_run: bool = False):
    """
    Espera a jogada terminar olhando o botão retrieve (retângulo roxo): enquanto
    ele está na tela as bolas ainda estão voando; quando some, a rodada acabou.

    Se o botão continuar na tela por mais de cfg.TEMPO_MAX_RODADA, as bolas
    provavelmente estão num loop quase horizontal — clica nele pra recolher as
    bolas e forçar o fim da rodada. Se o clique não resolver, tenta de novo a
    cada RECLIQUE_S, até MAX_CLIQUES; depois disso desiste e segue (o scan do
    próximo turno lida com o que estiver na tela).
    """
    POLL_S = 0.3  # intervalo entre checagens
    APARECER_S = 5.0  # quanto esperar o botão surgir depois do disparo
    RECLIQUE_S = 5.0  # intervalo entre cliques se o primeiro não encerrar
    MAX_CLIQUES = 3

    print("  Aguardando fim da rodada...")
    t0 = time.monotonic()

    # 1. O botão leva um instante pra aparecer depois do disparo. Sem essa
    #    espera, a primeira checagem veria "sem botão" e daria a rodada por
    #    encerrada antes dela começar.
    while not detect_retrieve_button():
        if time.monotonic() - t0 > APARECER_S:
            print("  [!] Botao retrieve nao apareceu — seguindo")
            return
        time.sleep(POLL_S)

    # 2. Espera o botão sumir, com limite de TEMPO_MAX_RODADA.
    cliques = 0
    proximo_clique = t0 + cfg.TEMPO_MAX_RODADA
    while detect_retrieve_button():
        agora = time.monotonic()
        if agora >= proximo_clique:
            if cliques >= MAX_CLIQUES:
                print(
                    f"  [!] Botao retrieve continua na tela apos {cliques} cliques — seguindo"
                )
                return
            cliques += 1
            print(
                f"  [!] Rodada passou de {cfg.TEMPO_MAX_RODADA:.0f}s — "
                f"clicando no retrieve pra forcar o fim ({cliques}/{MAX_CLIQUES})"
            )
            if not dry_run:
                click_retrieve_button()
            proximo_clique = agora + RECLIQUE_S
        time.sleep(POLL_S)

    print(f"  Rodada encerrada ({time.monotonic() - t0:.1f}s)")


def run_turn(
    state: GameState,
    debug: bool = False,
    dry_run: bool = False,
):
    """
    Executa um turno completo a partir de um GameState já capturado.
    Retorna status: True (sucesso), False (falha), "game_over"
    """
    if state.game_over:
        print(
            f"  [GAME OVER] Tela branca detectada ({state.white_ratio:.0%}) — parando bot"
        )
        return "game_over"

    if state.ball_x is None:
        print("  [!] Bola nao encontrada — pulando turno")
        return False

    if not state.blocks:
        print("  [!] Nenhum bloco detectado — possivelmente entre turnos")
        return False

    ball_x = state.ball_x
    ball_y = state.ball_y
    ball_count = max(1, state.detected_ball_count)

    print(f"  Bola: ({ball_x:.0f}, {ball_y:.0f})  |  Bolas: {ball_count}")
    print(f"  Blocos: {len(state.blocks)}  |  Power-ups: {len(state.powerups)}")

    # ── Estado por fileira (Level 1 = topo, Level 7 = fundo/perigo) ──
    levels_map = {i: [] for i in range(7)}
    for b in state.blocks:
        levels_map[b.row_index].append(b)

    for row_idx in range(6, -1, -1):
        display_level = 7 - row_idx
        blocks_in_lvl = levels_map[row_idx]
        if not blocks_in_lvl:
            continue
        entries = []
        for b in blocks_in_lvl:
            tipo = "Tri" if b.block_type.value == "triangle" else "Bloco"
            entries.append(f"{tipo} hp={b.hp}")
        alert = " [!!!] CRITICO" if row_idx == 0 else ""
        print(f"  Level {display_level}: [{', '.join(entries)}]{alert}")

    # ── 2. SIMULAÇÃO ──
    t0 = time.perf_counter()
    best, all_results = find_best_angle(
        ball_x,
        ball_y,
        state.blocks,
        state.powerups,
        ball_count=ball_count,
        progress_callback=make_progress_bar(),
    )
    sim_time = time.perf_counter() - t0
    update_shot_display(best, all_results, (ball_x, ball_y))

    print(f"  Melhor angulo: {best.angle:.1f}°")
    print(
        f"  Hits ponderados: {best.weighted_hits:.1f}  |  Danger hits: {best.danger_hits}  |  Total hits: {best.total_hits}"
    )
    print(
        f"  Score: {best.score:.0f} (destroyed={best.blocks_destroyed}, "
        f"powerups={best.powerups_collected}, +balls={best.extra_balls_collected})"
    )
    print(f"  Dano por hit: {ball_count}x  |  Simulacao: {sim_time:.2f}s")

    # ── 3. DEBUG VISUAL ──
    # A trajetoria agora e o painel "trajetoria" de visualizar_masks, desenhado
    # pela thread de masks. Antes esta funcao criava a propria janela, o que
    # punha DUAS threads chamando imshow/waitKey — o HighGUI do OpenCV nao e
    # thread-safe e travava a janela de forma intermitente no Windows.
    # (a flag --debug ficou sem uso; ligue/desligue o painel em JANELAS)

    # ── 4. EXECUÇÃO ──
    if not dry_run:
        execute_shot(ball_x, ball_y, best.angle)
        print("  >> Disparo executado!")
    else:
        print(f"  >> [DRY RUN] Tiro em {best.angle:.1f}° (nao executado)")

    # antigo: retornava um fingerprint das posicoes dos blocos, usado pela
    # deteccao de "jogo travado" (mesmo tabuleiro por 20 turnos). Removida: o
    # unico criterio pra forcar o fim de uma rodada agora e TEMPO_MAX_RODADA.
    return True


def main():
    args = parse_args()
    cfg.VISION_VERBOSE = args.verbose_vision

    if args.animate:
        if args.delay > 0:
            print(f"Iniciando em {args.delay:.0f}s — posicione o jogo!")
            time.sleep(args.delay)
        from tools.animate_dryrun import run_animate

        try:
            run_animate()
        finally:
            cv2.destroyAllWindows()
        return

    print("=" * 50)
    print("  BBTAN Bot")
    print("=" * 50)
    print(f"  Regiao do jogo: {cfg.GAME_REGION}")
    print(f"  Angulos: {cfg.ANGLE_MIN}° a {cfg.ANGLE_MAX}° (step={cfg.ANGLE_STEP}°)")
    print(f"  Debug: {args.debug}  |  Dry-run: {args.dry_run}")
    print(f"  Smart-wait: {args.smart_wait}")
    if args.prints:
        print(f"  Prints: salvando em {PRINTS_DIR}")
    print("=" * 50)
    print()

    masks_stop = threading.Event()
    masks_thread = threading.Thread(
        target=run_masks_window, args=(masks_stop,), daemon=True
    )
    masks_thread.start()

    if args.dry_run:
        print("Modo dry-run: apenas visualização. Ctrl+C para sair.")
        try:
            while True:
                state = scan_game()
                update_level_display(
                    state.debug_frame, state.blocks, state.detected_ball_count
                )
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nParado pelo usuario.")
        finally:
            masks_stop.set()
            masks_thread.join(timeout=2)
            cv2.destroyAllWindows()
        return

    if args.delay > 0:
        print(f"Iniciando em {args.delay:.0f}s — posicione o jogo!")
        time.sleep(args.delay)

    turn = 0
    consecutive_fails = 0
    max_fails = 5

    try:
        while True:
            turn += 1
            if args.max_turns > 0 and turn > args.max_turns:
                print(f"\nLimite de {args.max_turns} turnos atingido.")
                break

            print(f"\n--- Turno {turn} ---")

            # Captura e atualiza a janela "Grid: Blocos por fileira"
            time.sleep(3)
            state = scan_game()
            update_level_display(
                state.debug_frame, state.blocks, state.detected_ball_count
            )
            print("  [scan] Estado capturado — display atualizado")

            # Print da tela entre rodadas: é o mesmo frame que o bot analisou
            # neste turno (tabuleiro parado, antes da jogada).
            if args.prints:
                salvar_print(state.raw_frame, turn)

            status = run_turn(
                state=state,
                debug=args.debug,
                dry_run=args.dry_run,
            )

            if status == "game_over":
                turnos_sobrevividos = turn - 1
                print(f"\n{'='*50}")
                print(f"  GAME OVER — Score final apos {turnos_sobrevividos} turnos")
                print(f"  Bolas no ultimo turno: {state.detected_ball_count}")
                print(f"{'='*50}")

                with open("historico.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now():%Y-%m-%d %H:%M:%S} | Turnos: {turnos_sobrevividos} | Bolas: {state.detected_ball_count}\n"
                    )

                if not args.dry_run:
                    import pyautogui

                    for i in range(3):
                        print(f"  >> Clique restart {i+1}/3...")
                        pyautogui.click(1576, 781)
                        time.sleep(1)
                        pyautogui.click(1584, 838)
                        if i < 2:
                            time.sleep(4)

                turn = 0
                consecutive_fails = 0
                print("\n  Recomecando do zero!\n")
                continue

            if status:
                consecutive_fails = 0

                # Aguarda fim da rodada pelo botão retrieve (com limite de
                # TEMPO_MAX_RODADA — unico criterio pra forcar o fim; a antiga
                # deteccao de "mesmo tabuleiro por 20 turnos" foi removida).
                # antigo: polling das 4 regiões rosa
                # (detect_round_end) com timeout fixo de 30s, que seguia pro
                # próximo turno com as bolas ainda voando.
                aguardar_fim_da_rodada(dry_run=args.dry_run)

                # Espera os blocos descerem e estabilizarem
                time.sleep(2.0)
            else:
                consecutive_fails += 1
                if consecutive_fails >= max_fails:
                    print(f"\n[!] {max_fails} falhas consecutivas — abortando.")
                    print(
                        "    Verifique se o jogo esta visivel e a GAME_REGION esta correta."
                    )
                    break
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nBot parado pelo usuario (Ctrl+C)")

    finally:
        masks_stop.set()
        masks_thread.join(timeout=2)
        if args.debug:
            cv2.destroyAllWindows()

    print(f"\nTotal de turnos jogados: {turn - 1}")
    print("Ate a proxima!")


if __name__ == "__main__":
    main()
