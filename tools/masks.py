"""
masks.py — Visualizador de masks standalone (sem rodar o bot).

Mostra 4 janelas:
  1. Mask: Blocos (todas as cores)
  2. Mask: Contagem de bolas
  3. Mask: Fim de rodada
  4. Grid: Blocos por fileira

Controles: 'q' para sair | 'r' para forçar re-scan
"""

from visualizar_masks import run_masks_window

if __name__ == "__main__":
    print("=== VISUALIZADOR DE MASKS ===")
    print("'q' para sair | 'r' para forcar re-scan\n")
    run_masks_window()
