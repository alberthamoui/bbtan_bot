"""
main_vision.py — Loop de validacao do pipeline de visao do BBTAN Bot.

Roda apenas scan_game() em loop continuo e exibe o que esta sendo detectado:
  - Posicao e contagem da bola
  - Blocos detectados (tipo, HP, posicao por fileira)
  - Power-ups detectados
  - Estado de game_over (tela branca)

Nao executa fisica, simulacao, nem acoes no mouse.

Uso:
  python main_vision.py          -> Roda o loop de visao
  python main_vision.py --delay  -> Delay inicial antes de comecar (default: 1s)
"""

import argparse
import time
import threading
import cv2

from vision import scan_game, GameState
from tools.visualizar_masks import run_masks_window, update_level_display


def parse_args():
    parser = argparse.ArgumentParser(description="BBTAN Bot — Vision Only")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay inicial em segundos antes de comecar (default: 1.0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Intervalo entre cada scan em segundos (default: 1.0)",
    )
    return parser.parse_args()


def print_state(state: GameState, scan_num: int):
    """Imprime no terminal o estado detectado pelo scan_game()."""
    print(f"\n--- Scan #{scan_num} ---")

    if state.game_over:
        print(f"  [GAME OVER] Tela branca detectada ({state.white_ratio:.0%})")
        return

    if state.ball_x is None:
        print("  [!] Bola nao encontrada")
    else:
        print(f"  Bola: ({state.ball_x:.0f}, {state.ball_y:.0f})  |  Bolas detectadas: {state.detected_ball_count}")

    if not state.blocks:
        print("  [!] Nenhum bloco detectado")
    else:
        print(f"  Blocos: {len(state.blocks)}  |  Power-ups: {len(state.powerups)}")

        # Agrupa blocos por fileira (row_index 0 = topo, 6 = fundo)
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

    if state.powerups:
        for p in state.powerups:
            print(f"  Power-up: tipo={p.power_up_type.value}  pos=({p.cx:.0f}, {p.cy:.0f})")


def main():
    args = parse_args()

    print("=" * 50)
    print("  BBTAN Bot — Vision Only")
    print("=" * 50)
    print(f"  Delay inicial: {args.delay}s  |  Intervalo: {args.interval}s")
    print("  Ctrl+C para sair")
    print("=" * 50)

    # Inicia janela de masks em thread separada
    masks_stop = threading.Event()
    masks_thread = threading.Thread(target=run_masks_window, args=(masks_stop,), daemon=True)
    masks_thread.start()

    if args.delay > 0:
        print(f"\nIniciando em {args.delay:.0f}s — posicione o jogo!")
        time.sleep(args.delay)

    scan_num = 0

    try:
        while True:
            scan_num += 1
            state = scan_game()
            update_level_display(state.debug_frame, state.blocks, state.detected_ball_count)
            print_state(state, scan_num)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nLoop de visao parado pelo usuario (Ctrl+C)")

    finally:
        masks_stop.set()
        masks_thread.join(timeout=2)
        cv2.destroyAllWindows()

    print(f"\nTotal de scans realizados: {scan_num}")


if __name__ == "__main__":
    main()