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
"""

import argparse
import time
import threading
import cv2

import config as cfg
from vision import scan_game, detect_round_end, GameState
from physics import find_best_angle
from executor import execute_shot
from tools.debug_view import draw_trajectory, draw_stats_overlay
from tools.visualizar_masks import run_masks_window, update_level_display


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
    return parser.parse_args()


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
    )
    sim_time = time.perf_counter() - t0

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
    if debug:
        vis = state.debug_frame.copy()
        vis = draw_trajectory(vis, best.path, color=(0, 200, 255), thickness=2)
        vis = draw_stats_overlay(vis, best, state)
        cv2.imshow("Bot: Trajetoria escolhida", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False, 0

    # ── 4. EXECUÇÃO ──
    if not dry_run:
        execute_shot(ball_x, ball_y, best.angle)
        print("  >> Disparo executado!")
    else:
        print(f"  >> [DRY RUN] Tiro em {best.angle:.1f}° (nao executado)")

    block_fingerprint = frozenset(
        (b.row_index, round(b.cx / cfg.BLOCK_SIZE)) for b in state.blocks
    )
    return block_fingerprint


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
    last_block_fingerprint = None
    stale_turns = 0
    max_stale_turns = 20

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

                from datetime import datetime

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
                last_block_fingerprint = None
                stale_turns = 0
                print("\n  Recomecando do zero!\n")
                continue

            if status:
                consecutive_fails = 0

                # Detecta jogo travado (blocos na mesma posicao por N turnos)
                if status == last_block_fingerprint:
                    stale_turns += 1
                else:
                    stale_turns = 0
                    last_block_fingerprint = status

                if stale_turns >= max_stale_turns:
                    print(
                        f"\n  [!] Blocos na mesma posicao por {max_stale_turns} turnos — clicando para destravar bolas"
                    )
                    if not args.dry_run:
                        import pyautogui

                        pyautogui.click(1580, 961)
                    stale_turns = 0

                # Aguarda fim da rodada (polling visual das 4 regiões rosa)
                print("  Aguardando fim da rodada...")
                timeout = 30.0
                elapsed = 0.0
                while elapsed < timeout:
                    if detect_round_end():
                        print("  Rodada encerrada detectada!")
                        break
                    time.sleep(0.3)
                    elapsed += 0.3
                else:
                    print("  [!] Timeout esperando fim da rodada")

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
