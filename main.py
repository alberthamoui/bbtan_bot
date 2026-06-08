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
import sys
import cv2

import config as cfg
from vision import scan_game, capture_screen, count_balls, GameState
from physics import find_best_angle, SimResult
from executor import execute_shot, wait_for_turn_end, wait_for_turn_end_smart
from tools.debug_view import draw_trajectory, draw_stats_overlay


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
        "--balls",
        type=int,
        default=0,
        help="Override: numero fixo de bolas (0 = detectar visualmente)",
    )
    return parser.parse_args()


def run_turn(
    ball_count_override: int = 0,
    current_ball_count: int = 1,
    debug: bool = False,
    dry_run: bool = False,
) -> tuple:
    """
    Executa um turno completo.
    Retorna (status, extra_balls_predicted)
    status: True (sucesso), False (falha), "game_over"
    extra_balls_predicted: power-ups (+) previstos pela simulação
    """
    # ── 1. CAPTURA E DETECÇÃO ──
    state = scan_game()

    if state.game_over:
        print(
            f"  [GAME OVER] Tela branca detectada ({state.white_ratio:.0%}) — parando bot"
        )
        return "game_over", 0

    if state.ball_x is None:
        print("  [!] Bola nao encontrada — pulando turno")
        return False, 0

    if not state.blocks:
        print("  [!] Nenhum bloco detectado — possivelmente entre turnos")
        return False, 0

    ball_x = state.ball_x
    ball_y = state.ball_y
    if ball_count_override > 0:
        ball_count = ball_count_override
    else:
        ball_count = max(state.detected_ball_count, current_ball_count)

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
        f"  Score: {best.score:.0f} (hits={best.total_hits}, "
        f"destroyed={best.blocks_destroyed}, powerups={best.powerups_collected}, "
        f"+balls={best.extra_balls_collected})"
    )
    print(f"  Dano por hit: {ball_count}x  |  Simulacao: {sim_time:.2f}s")

    # ── 3. DEBUG VISUAL ──
    if debug:
        vis = state.debug_frame.copy()
        vis = draw_trajectory(vis, best.path, color=(0, 200, 255), thickness=2)
        vis = draw_stats_overlay(vis, best, state)
        cv2.imshow("BBTAN Bot", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False, 0

    # ── 4. EXECUÇÃO ──
    if not dry_run:
        execute_shot(ball_x, ball_y, best.angle)
        print(f"  >> Disparo executado!")
    else:
        print(f"  >> [DRY RUN] Tiro em {best.angle:.1f}° (nao executado)")

    return True, best.extra_balls_collected


def main():
    args = parse_args()

    print("=" * 50)
    print("  BBTAN Bot")
    print("=" * 50)
    print(f"  Regiao do jogo: {cfg.GAME_REGION}")
    print(f"  Angulos: {cfg.ANGLE_MIN}° a {cfg.ANGLE_MAX}° (step={cfg.ANGLE_STEP}°)")
    print(f"  Debug: {args.debug}  |  Dry-run: {args.dry_run}")
    print(f"  Smart-wait: {args.smart_wait}")
    print("=" * 50)
    print()

    if args.delay > 0:
        print(f"Iniciando em {args.delay:.0f}s — posicione o jogo!")
        time.sleep(args.delay)

    turn = 0
    ball_count = 1
    consecutive_fails = 0
    max_fails = 5

    try:
        while True:
            turn += 1
            if args.max_turns > 0 and turn > args.max_turns:
                print(f"\nLimite de {args.max_turns} turnos atingido.")
                break

            print(f"\n--- Turno {turn} ---")

            status, extra_balls = run_turn(
                ball_count_override=args.balls,
                current_ball_count=ball_count,
                debug=args.debug,
                dry_run=args.dry_run,
            )

            if status == "game_over":
                print(f"\n{'='*50}")
                print(f"  GAME OVER — Score final apos {turn - 1} turnos")
                print(f"  Bolas no ultimo turno: {ball_count}")
                print(f"{'='*50}")
                break

            if status:
                consecutive_fails = 0

                # Acumula bolas previstas pela simulação
                if args.balls > 0:
                    ball_count = args.balls
                elif extra_balls > 0:
                    ball_count += extra_balls
                    print(f"  [+] +{extra_balls} bolas previstas → total: {ball_count}")

                # Espera o turno acabar
                if args.smart_wait:
                    print("  Aguardando turno (deteccao visual)...")
                    wait_for_turn_end_smart()
                else:
                    print(f"  Aguardando {cfg.TURN_WAIT_TIME}s...")
                    wait_for_turn_end()

                # Recontagem visual pos-turno (bolas agrupadas na base)
                if args.balls <= 0:
                    post_frame = capture_screen()
                    visual_count = count_balls(post_frame)
                    if visual_count > ball_count:
                        print(f"  [+] Recontagem visual: {visual_count} bolas")
                        ball_count = visual_count
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
        if args.debug:
            cv2.destroyAllWindows()

    print(f"\nTotal de turnos jogados: {turn - 1}")
    print("Ate a proxima!")


if __name__ == "__main__":
    main()
