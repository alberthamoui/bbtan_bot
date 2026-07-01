"""
capture_hp_templates.py — Captura interativa de templates de dígitos do HP.

Roda scan_game() em loop. Para cada bloco detectado, pega cada componente
conectado válido da hp_mask (mesma logica de _read_hp) e exibe ampliado.
Você digita o dígito direto na janela do OpenCV (foco precisa estar nela).

Salva em templates/hp_digits/<digito>_<timestamp>.png — um arquivo por
amostra; depois usa-se média/voto pra construir o template final.

Uso:
  python capture_hp_templates.py
  python capture_hp_templates.py --delay 3   # delay inicial pra posicionar o jogo

Teclas (foco na janela "Template Capture"):
  0-9 = salva esse crop como aquele dígito
  s   = skip esse componente
  n   = pula resto desse bloco
  f   = pula resto desse frame (captura novo)
  q   = sair
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision import scan_game  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "templates" / "hp_digits"
ZOOM = 10
WIN = "Template Capture"


def parse_args():
    p = argparse.ArgumentParser(description="Captura de templates HP")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Delay inicial em segundos (default: 1.0)")
    return p.parse_args()


def _render(crop: np.ndarray, info: str) -> np.ndarray:
    """Constrói a imagem ampliada pra exibir na janela."""
    h, w = crop.shape[:2]
    disp = cv2.resize(crop, (max(w * ZOOM, 200), max(h * ZOOM, 100)),
                      interpolation=cv2.INTER_NEAREST)
    disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    # Faixa preta embaixo pra texto
    bar = np.zeros((40, disp_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, info, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 220, 0), 1)
    cv2.putText(bar, "0-9 salva | s skip | n bloco | f frame | q sair",
                (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return np.vstack([disp_bgr, bar])


def _wait_key() -> str:
    """Bloqueia no waitKey mantendo o event loop vivo. Retorna 'q','s','n','f',ou dígito."""
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 0xFF or key == 0:  # sem tecla / modificador
            continue
        ch = chr(key) if 32 <= key < 127 else ""
        if ch in ("q", "s", "n", "f"):
            return ch
        if ch.isdigit():
            return ch
        # tecla ignorada — continua esperando


def _components(hp_mask: np.ndarray):
    """Mesma lógica de filtro que _read_hp aplica."""
    if hp_mask.size == 0:
        return []
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(hp_mask, connectivity=8)
    min_area = max(3, hp_mask.shape[0] * hp_mask.shape[1] // 100)
    comps = []
    for i in range(1, n_labels):
        cx, cy, cw, ch, ca = stats[i]
        if ca < min_area or ch < 4:
            continue
        crop = hp_mask[cy:cy + ch, cx:cx + cw]
        comps.append((cx, crop))
    comps.sort(key=lambda c: c[0])
    return comps


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Captura de templates HP")
    print("=" * 60)
    print(f"  Saída: {OUT_DIR}")
    print("  FOCO na janela 'Template Capture' pra teclas funcionarem.")
    print("  0-9 = salvar | s = skip | n = bloco | f = frame | q = sair")
    print("=" * 60)

    if args.delay > 0:
        print(f"\nIniciando em {args.delay:.0f}s — posicione o jogo.")
        time.sleep(args.delay)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    seen = set()
    saved_count = {str(d): 0 for d in range(10)}

    try:
        while True:
            print("\n— novo frame —")
            state = scan_game()

            if state.game_over:
                print("  GAME OVER detectado, aguardando 2s...")
                time.sleep(2)
                continue
            if not state.blocks:
                print("  Sem blocos. Aguardando 2s...")
                time.sleep(2)
                continue

            skip_frame = False
            for b in state.blocks:
                comps = _components(b.hp_mask)
                if not comps:
                    continue

                tipo = "TRI" if b.block_type.value == "triangle" else "BLK"
                print(f"\n  {tipo} L{7 - b.row_index}C{b.col_index} "
                      f"hp_lido={b.hp}  ({len(comps)} componente(s))")

                skip_block = False
                for ci, (_, crop) in enumerate(comps):
                    sig = (crop.shape, crop.tobytes())
                    if sig in seen:
                        print(f"    [{ci + 1}/{len(comps)}] já visto, skip")
                        continue

                    info = (f"{tipo} L{7 - b.row_index}C{b.col_index}  "
                            f"{crop.shape[1]}x{crop.shape[0]}  "
                            f"({ci + 1}/{len(comps)})")
                    img = _render(crop, info)
                    cv2.imshow(WIN, img)
                    print(f"    [{ci + 1}/{len(comps)}] shape={crop.shape} "
                          f"— pressione tecla NA JANELA")

                    resp = _wait_key()

                    if resp == "q":
                        raise KeyboardInterrupt
                    if resp == "f":
                        skip_frame = True
                        break
                    if resp == "n":
                        skip_block = True
                        break
                    if resp == "s":
                        continue
                    if resp.isdigit():
                        ts = int(time.time() * 1000)
                        out_path = OUT_DIR / f"{resp}_{ts}.png"
                        cv2.imwrite(str(out_path), crop)
                        seen.add(sig)
                        saved_count[resp] += 1
                        print(f"      salvo {out_path.name}  "
                              f"(total '{resp}': {saved_count[resp]})")

                if skip_block:
                    continue
                if skip_frame:
                    break

            if skip_frame:
                continue

            # Acabaram os componentes do frame — espera enter na janela
            blank = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(blank, "Frame revisado.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1)
            cv2.putText(blank, "qualquer tecla = proximo | q = sair",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1)
            cv2.imshow(WIN, blank)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n\nInterrompido.")
    finally:
        cv2.destroyAllWindows()

    print("\n=== Resumo ===")
    total = 0
    for d, n in saved_count.items():
        if n > 0:
            print(f"  {d}: {n} amostras")
            total += n
    print(f"Total: {total} amostras")
    print(f"Saída: {OUT_DIR}")


if __name__ == "__main__":
    main()
