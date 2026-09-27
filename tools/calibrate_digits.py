"""
calibrate_digits.py — Captura templates dos dígitos 0-9 do contador de bolas.

Painel triplo:
  [ORIGINAL]  imagem BGR do componente
  [MASK]      mask binária extraída
  [PREVIEW]   como o template ficará salvo (altura 28px, binarizado)

Auto-sugere o dígito via matching contra os templates já salvos.

Controles:
  0-9   salva componente como template do dígito
  X     pula sem salvar
  B     volta pro componente anterior
  D     deleta template existente (do dígito mostrado na sugestão)
  R     recaptura a tela
  Q     sai

Aperta o mesmo dígito duas vezes pra sobrescrever um template existente.

Templates salvos em templates/digits/
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mss
import config as cfg
from vision import _match_digit, _load_digit_templates, _DIGIT_TEMPLATES

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "digits"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

DIGIT_H = 28
WIN_NAME = "Calibrar: Templates de digitos (bolas)"

CELL_W = 220
CELL_H = 170
SEP_COLOR = (60, 60, 60)


def capture_components():
    """Captura BALL_COUNT_REGION, gera mask e separa componentes."""
    sct = mss.mss()
    bx, by, bw, bh = cfg.BALL_COUNT_REGION
    mon = {"top": by, "left": bx, "width": bw, "height": bh}
    img_bgr = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.BALL_COUNT_HSV_LOW, cfg.BALL_COUNT_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < 15 or h < 6:
            continue
        components.append(
            {
                "bbox": (int(x), int(y), int(w), int(h)),
                "mask_crop": mask[y : y + h, x : x + w].copy(),
                "bgr_crop": img_bgr[y : y + h, x : x + w].copy(),
            }
        )

    components.sort(key=lambda c: c["bbox"][0])
    return img_bgr, mask, components


def make_preview(crop_mask: np.ndarray) -> np.ndarray:
    """Renderiza o template como será salvo: altura DIGIT_H, binarizado."""
    aspect = crop_mask.shape[1] / max(crop_mask.shape[0], 1)
    w_norm = max(1, round(DIGIT_H * aspect))
    resized = cv2.resize(crop_mask, (w_norm, DIGIT_H), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def save_template(digit: str, crop_mask: np.ndarray):
    preview = make_preview(crop_mask)
    path = TEMPLATES_DIR / f"{digit}.png"
    cv2.imwrite(str(path), preview)
    # Atualiza cache em memória pra sugestão refletir mudança imediata
    _DIGIT_TEMPLATES[digit] = preview
    return path, preview.shape[1]


def delete_template(digit: str) -> bool:
    path = TEMPLATES_DIR / f"{digit}.png"
    if path.exists():
        path.unlink()
        _DIGIT_TEMPLATES.pop(digit, None)
        return True
    return False


def suggest_digit(crop_mask: np.ndarray):
    if not _DIGIT_TEMPLATES:
        return None
    return _match_digit(crop_mask)


def load_saved_state() -> dict:
    saved = {}
    for p in TEMPLATES_DIR.glob("*.png"):
        if p.stem.isdigit():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 10:
                saved[p.stem] = True
    return saved


def fit_panel(img, target_w: int, target_h: int) -> np.ndarray:
    """Centraliza img num canvas de tamanho fixo, escalando proporcional."""
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if img is None or img.size == 0:
        return canvas
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas


def pad_to_width(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] == w:
        return img
    if img.shape[1] > w:
        return img[:, :w]
    pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
    return np.hstack([img, pad])


def build_overview(mask: np.ndarray, components, current_idx: int) -> np.ndarray:
    scale = 2
    mask_big = cv2.resize(
        mask,
        (mask.shape[1] * scale, mask.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    vis = cv2.cvtColor(mask_big, cv2.COLOR_GRAY2BGR)
    for i, comp in enumerate(components):
        x, y, w, h = comp["bbox"]
        color = (0, 255, 0) if i == current_idx else (80, 80, 80)
        thickness = 2 if i == current_idx else 1
        cv2.rectangle(
            vis,
            (x * scale, y * scale),
            ((x + w) * scale, (y + h) * scale),
            color,
            thickness,
        )
    return vis


def build_triple_panel(comp, sugg) -> np.ndarray:
    """Painel triplo: original | mask | preview do template."""
    panel_bgr = fit_panel(comp["bgr_crop"], CELL_W, CELL_H)
    panel_mask = fit_panel(comp["mask_crop"], CELL_W, CELL_H)
    panel_preview = fit_panel(make_preview(comp["mask_crop"]), CELL_W, CELL_H)

    sep = np.full((CELL_H, 3, 3), SEP_COLOR, dtype=np.uint8)
    triple = np.hstack([panel_bgr, sep, panel_mask, sep, panel_preview])

    # Labels acima de cada painel
    labels = np.zeros((22, triple.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        labels,
        "ORIGINAL (BGR)",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        labels,
        "MASK",
        (CELL_W + 11, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        labels,
        f"PREVIEW ({DIGIT_H}px)",
        (2 * CELL_W + 14, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    # Caixa de sugestão centralizada embaixo
    sugg_h = 36
    sugg_bar = np.zeros((sugg_h, triple.shape[1], 3), dtype=np.uint8)
    if sugg is not None:
        text = f"Sugestao: {sugg}"
        color = (0, 220, 255)
    else:
        text = "Sugestao: -"
        color = (120, 120, 120)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        sugg_bar,
        text,
        ((triple.shape[1] - tw) // 2, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return np.vstack([labels, triple, sugg_bar])


def build_status_bar(
    width: int, current_idx: int, n: int, saved: dict, message: str, armed_overwrite
) -> np.ndarray:
    info_h = 80
    info = np.zeros((info_h, width, 3), dtype=np.uint8)

    idx_text = f"Componente {min(current_idx, n - 1) + 1 if n else 0}/{n}"
    cv2.putText(
        info, idx_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    # Status dos templates (0-9) em verde/cinza
    x = 8
    cv2.putText(
        info, "Templates:", (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
    )
    x = 110
    for d in range(10):
        color = (100, 230, 100) if saved.get(str(d)) else (90, 90, 90)
        cv2.putText(info, f"{d}", (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        x += 22

    cv2.putText(
        info,
        "0-9: salvar  X: pular  B: voltar  D: deletar  R: recapturar  Q: sair",
        (8, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
    )

    msg_color = (0, 220, 255)
    if armed_overwrite is not None:
        cv2.putText(
            info,
            f"AVISO: '{armed_overwrite}' ja existe. Aperte de novo para sobrescrever.",
            (8, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 180, 255),
            1,
        )
    elif message:
        cv2.putText(
            info, message, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, msg_color, 1
        )

    return info


def build_display(mask, components, current_idx, saved, message, armed_overwrite):
    if 0 <= current_idx < len(components):
        comp = components[current_idx]
        sugg = suggest_digit(comp["mask_crop"])
        triple = build_triple_panel(comp, sugg)
    else:
        triple = np.zeros((CELL_H + 60, 3 * CELL_W + 6, 3), dtype=np.uint8)
        cv2.putText(
            triple,
            "Sem componentes. R=recapturar  Q=sair",
            (10, CELL_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

    overview = build_overview(mask, components, current_idx)
    status = build_status_bar(
        triple.shape[1], current_idx, len(components), saved, message, armed_overwrite
    )

    target_w = max(overview.shape[1], triple.shape[1], status.shape[1])
    overview = pad_to_width(overview, target_w)
    triple = pad_to_width(triple, target_w)
    status = pad_to_width(status, target_w)

    return np.vstack([overview, triple, status])


def main():
    print("=== CALIBRAR TEMPLATES DE DIGITOS (BALL COUNT) ===")
    print("Janela: 0-9 salva, X pula, B volta, D deleta, R recaptura, Q sai\n")

    _load_digit_templates()
    saved = load_saved_state()

    img_bgr, mask, components = capture_components()
    current_idx = 0
    message = f"Encontrados {len(components)} componentes"
    msg_set_at = time.time()
    armed_overwrite = None

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        # Auto-clear mensagem após 2.5s
        if message and time.time() - msg_set_at > 2.5:
            message = ""

        display = build_display(
            mask, components, current_idx, saved, message, armed_overwrite
        )
        cv2.imshow(WIN_NAME, display)
        key = cv2.waitKey(80) & 0xFF

        if key == 255:
            continue

        if key == ord("q"):
            break

        if key == ord("r"):
            img_bgr, mask, components = capture_components()
            current_idx = 0
            armed_overwrite = None
            message = f"Recapturado: {len(components)} componentes"
            msg_set_at = time.time()
            continue

        if key == ord("x"):
            if current_idx < len(components):
                current_idx += 1
                armed_overwrite = None
                if current_idx >= len(components):
                    message = "Todos processados. R=recapturar  Q=sair"
                    msg_set_at = time.time()
            continue

        if key == ord("b"):
            if current_idx > 0:
                current_idx -= 1
                armed_overwrite = None
                message = f"Voltou para componente {current_idx + 1}"
                msg_set_at = time.time()
            continue

        if key == ord("d"):
            if 0 <= current_idx < len(components):
                sugg = suggest_digit(components[current_idx]["mask_crop"])
                if sugg is not None and delete_template(sugg):
                    saved.pop(sugg, None)
                    message = f"Deletado: '{sugg}.png'"
                else:
                    message = "Nada a deletar (sem sugestao ou arquivo)"
                msg_set_at = time.time()
                armed_overwrite = None
            continue

        if ord("0") <= key <= ord("9") and current_idx < len(components):
            digit = chr(key)
            already = (TEMPLATES_DIR / f"{digit}.png").exists()

            if already and armed_overwrite != digit:
                armed_overwrite = digit
                message = ""
                continue

            crop = components[current_idx]["mask_crop"]
            path, w_norm = save_template(digit, crop)
            saved[digit] = True
            armed_overwrite = None
            message = f"Salvo: '{digit}' -> {path.name} ({w_norm}x{DIGIT_H})"
            msg_set_at = time.time()
            current_idx += 1
            if current_idx >= len(components):
                message += " | Todos processados"
            continue

        # Qualquer outra tecla desarma overwrite
        armed_overwrite = None

    cv2.destroyAllWindows()
    print("\nResumo dos templates:")
    for d in range(10):
        status = "OK" if saved.get(str(d)) else "FALTA"
        print(f"  {d}: {status}")
    print(f"\nDiretorio: {TEMPLATES_DIR}")


if __name__ == "__main__":
    main()
