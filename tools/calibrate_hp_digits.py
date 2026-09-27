"""
calibrate_hp_digits.py — Captura templates dos dígitos de HP dentro dos blocos.

Captura a região do jogo (GAME_REGION), detecta os blocos e usa
_get_block_text_mask de vision.py para isolar o texto dentro de cada bloco.
Os componentes (dígitos) são extraídos via connectedComponents.

Painel:
  [BLOCO]       ROI BGR do bloco com retângulo do componente atual
  [TEXT MASK]   text_mask do bloco inteiro (saída de _get_block_text_mask)
  [ORIGINAL]    BGR só do componente atual
  [MASK]        mask binária só do componente
  [PREVIEW]     como o template ficará salvo (altura 28px, binarizado)

Auto-sugere o dígito via matching contra os templates já salvos em
templates/digits/ (compartilhado com calibrate_digits.py).

Controles:
  0-9   salva componente como template do dígito
  X     pula sem salvar (próximo componente)
  B     volta pro componente anterior
  N     pula direto pro próximo bloco
  D     deleta template existente (do dígito mostrado na sugestão)
  R     recaptura a tela
  Q     sai

Aperta o mesmo dígito duas vezes pra sobrescrever um template existente.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mss
import config as cfg
from vision import (
    _match_digit,
    _load_digit_templates,
    _DIGIT_TEMPLATES,
    _get_block_text_mask,
    detect_blocks,
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "digits"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

DIGIT_H = 28
WIN_NAME = "Calibrar: Templates de digitos (HP)"

CELL_W = 200
CELL_H = 160
SEP_COLOR = (60, 60, 60)


def capture_frame():
    sct = mss.mss()
    rx, ry, rw, rh = cfg.GAME_REGION
    mon = {"top": ry, "left": rx, "width": rw, "height": rh}
    return np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]


def extract_components_from_blocks(frame: np.ndarray):
    """
    Para cada bloco detectado, gera o text_mask via _get_block_text_mask,
    separa via connectedComponents e retorna lista plana de componentes,
    cada um com referência ao bloco pai.
    """
    blocks = detect_blocks(frame)
    components_list = []

    for bi, b in enumerate(blocks):
        # Aplica o mesmo padding interno usado em _read_hp
        pad_x = max(2, b.w // 6)
        pad_y = max(2, b.h // 6)
        x1, y1 = b.x + pad_x, b.y + pad_y
        x2, y2 = b.x + b.w - pad_x, b.y + b.h - pad_y
        if x2 <= x1 or y2 <= y1:
            continue
        roi_bgr = frame[y1:y2, x1:x2]
        if roi_bgr.size == 0 or roi_bgr.shape[0] < 4 or roi_bgr.shape[1] < 4:
            continue

        text_mask = _get_block_text_mask(roi_bgr)
        if np.count_nonzero(text_mask) < 5:
            continue

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            text_mask, connectivity=8
        )
        min_area = max(3, (x2 - x1) * (y2 - y1) // 100)

        comps_in_block = []
        for i in range(1, n_labels):
            cx, cy, cw, ch, ca = stats[i]
            if ca < min_area or ch < 4:
                continue
            comps_in_block.append(
                {
                    "comp_bbox": (int(cx), int(cy), int(cw), int(ch)),
                    "mask_crop": text_mask[cy : cy + ch, cx : cx + cw].copy(),
                    "bgr_crop": roi_bgr[cy : cy + ch, cx : cx + cw].copy(),
                }
            )

        if not comps_in_block:
            continue

        # Ordena por x (esquerda → direita, ordem de leitura)
        comps_in_block.sort(key=lambda c: c["comp_bbox"][0])

        # Anexa info do bloco pai a cada componente
        for ci, comp in enumerate(comps_in_block):
            components_list.append(
                {
                    "block_idx": bi,
                    "block_label": f"L{7 - b.row_index}C{b.col_index}",
                    "block_bbox": (b.x, b.y, b.w, b.h),
                    "block_bgr": frame[b.y : b.y + b.h, b.x : b.x + b.w].copy(),
                    "block_text_mask": text_mask.copy(),
                    "roi_origin": (x1 - b.x, y1 - b.y),  # offset do roi dentro do bloco
                    "comp_idx_in_block": ci,
                    "comps_in_block": len(comps_in_block),
                    "comp_bbox": comp["comp_bbox"],
                    "mask_crop": comp["mask_crop"],
                    "bgr_crop": comp["bgr_crop"],
                }
            )

    return blocks, components_list


def make_preview(crop_mask: np.ndarray) -> np.ndarray:
    aspect = crop_mask.shape[1] / max(crop_mask.shape[0], 1)
    w_norm = max(1, round(DIGIT_H * aspect))
    resized = cv2.resize(crop_mask, (w_norm, DIGIT_H), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def save_template(digit: str, crop_mask: np.ndarray):
    preview = make_preview(crop_mask)
    path = TEMPLATES_DIR / f"{digit}.png"
    cv2.imwrite(str(path), preview)
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


def build_overview(frame: np.ndarray, blocks, current_block_idx: int) -> np.ndarray:
    """Mini-vista do frame inteiro com highlight do bloco atual."""
    scale = 0.5
    h, w = frame.shape[:2]
    mini = cv2.resize(
        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    for i, b in enumerate(blocks):
        x1 = int(b.x * scale)
        y1 = int(b.y * scale)
        x2 = int((b.x + b.w) * scale)
        y2 = int((b.y + b.h) * scale)
        color = (0, 255, 0) if i == current_block_idx else (80, 80, 80)
        thickness = 2 if i == current_block_idx else 1
        cv2.rectangle(mini, (x1, y1), (x2, y2), color, thickness)
    return mini


def build_block_panel(comp) -> np.ndarray:
    """Painel duplo: bloco BGR (com highlight do componente) | text_mask."""
    block_bgr = comp["block_bgr"].copy()
    ox, oy = comp["roi_origin"]
    cx, cy, cw, ch = comp["comp_bbox"]
    # Coord do componente dentro do bloco
    rx1 = ox + cx
    ry1 = oy + cy
    rx2 = rx1 + cw
    ry2 = ry1 + ch
    cv2.rectangle(block_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

    panel_block = fit_panel(block_bgr, CELL_W, CELL_H)
    panel_mask = fit_panel(comp["block_text_mask"], CELL_W, CELL_H)

    sep = np.full((CELL_H, 3, 3), SEP_COLOR, dtype=np.uint8)
    pair = np.hstack([panel_block, sep, panel_mask])

    labels = np.zeros((22, pair.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        labels,
        f"BLOCO {comp['block_label']}",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        labels,
        "TEXT MASK",
        (CELL_W + 11, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    return np.vstack([labels, pair])


def build_comp_panel(comp, sugg) -> np.ndarray:
    """Painel triplo: BGR do componente | mask | preview."""
    panel_bgr = fit_panel(comp["bgr_crop"], CELL_W, CELL_H)
    panel_mask = fit_panel(comp["mask_crop"], CELL_W, CELL_H)
    panel_preview = fit_panel(make_preview(comp["mask_crop"]), CELL_W, CELL_H)

    sep = np.full((CELL_H, 3, 3), SEP_COLOR, dtype=np.uint8)
    triple = np.hstack([panel_bgr, sep, panel_mask, sep, panel_preview])

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
    width: int,
    comp,
    current_idx: int,
    n: int,
    saved: dict,
    message: str,
    armed_overwrite,
) -> np.ndarray:
    info_h = 80
    info = np.zeros((info_h, width, 3), dtype=np.uint8)

    if comp is not None:
        idx_text = (
            f"Componente {current_idx + 1}/{n}  |  "
            f"Digito {comp['comp_idx_in_block'] + 1}/{comp['comps_in_block']} "
            f"do bloco {comp['block_label']}"
        )
    else:
        idx_text = f"Componente 0/{n}"
    cv2.putText(
        info, idx_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    cv2.putText(
        info, "Templates:", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
    )
    x = 110
    for d in range(10):
        color = (100, 230, 100) if saved.get(str(d)) else (90, 90, 90)
        cv2.putText(info, f"{d}", (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        x += 22

    cv2.putText(
        info,
        "0-9: salvar  X: pular  B: voltar  N: prox bloco  D: deletar  R: recapturar  Q: sair",
        (8, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (150, 150, 150),
        1,
    )

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
            info, message, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1
        )

    return info


def build_display(
    frame, blocks, components_list, current_idx, saved, message, armed_overwrite
):
    if 0 <= current_idx < len(components_list):
        comp = components_list[current_idx]
        sugg = suggest_digit(comp["mask_crop"])
        block_panel = build_block_panel(comp)
        comp_panel = build_comp_panel(comp, sugg)
        current_block_idx = comp["block_idx"]
    else:
        comp = None
        block_panel = np.zeros((CELL_H + 22, 2 * CELL_W + 3, 3), dtype=np.uint8)
        comp_panel = np.zeros((CELL_H + 60, 3 * CELL_W + 6, 3), dtype=np.uint8)
        cv2.putText(
            comp_panel,
            "Sem componentes. R=recapturar  Q=sair",
            (10, CELL_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )
        current_block_idx = -1

    overview = build_overview(frame, blocks, current_block_idx)
    status = build_status_bar(
        max(block_panel.shape[1], comp_panel.shape[1]),
        comp,
        current_idx,
        len(components_list),
        saved,
        message,
        armed_overwrite,
    )

    target_w = max(
        overview.shape[1], block_panel.shape[1], comp_panel.shape[1], status.shape[1]
    )
    overview = pad_to_width(overview, target_w)
    block_panel = pad_to_width(block_panel, target_w)
    comp_panel = pad_to_width(comp_panel, target_w)
    status = pad_to_width(status, target_w)

    return np.vstack([overview, block_panel, comp_panel, status])


def next_block(components_list, current_idx: int) -> int:
    """Pula até o primeiro componente do próximo bloco."""
    if current_idx >= len(components_list):
        return current_idx
    cur_block = components_list[current_idx]["block_idx"]
    for i in range(current_idx + 1, len(components_list)):
        if components_list[i]["block_idx"] != cur_block:
            return i
    return len(components_list)


def main():
    print("=== CALIBRAR TEMPLATES DE HP DIGITS (DENTRO DOS BLOCOS) ===")
    print(
        "Janela: 0-9 salva, X pula, B volta, N prox bloco, D deleta, R recaptura, Q sai\n"
    )

    _load_digit_templates()
    saved = load_saved_state()

    frame = capture_frame()
    blocks, components_list = extract_components_from_blocks(frame)
    current_idx = 0
    message = f"{len(blocks)} blocos, {len(components_list)} componentes encontrados"
    msg_set_at = time.time()
    armed_overwrite = None

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        if message and time.time() - msg_set_at > 2.5:
            message = ""

        display = build_display(
            frame, blocks, components_list, current_idx, saved, message, armed_overwrite
        )
        cv2.imshow(WIN_NAME, display)
        key = cv2.waitKey(80) & 0xFF

        if key == 255:
            continue

        if key == ord("q"):
            break

        if key == ord("r"):
            frame = capture_frame()
            blocks, components_list = extract_components_from_blocks(frame)
            current_idx = 0
            armed_overwrite = None
            message = (
                f"Recapturado: {len(blocks)} blocos, {len(components_list)} componentes"
            )
            msg_set_at = time.time()
            continue

        if key == ord("x"):
            if current_idx < len(components_list):
                current_idx += 1
                armed_overwrite = None
                if current_idx >= len(components_list):
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

        if key == ord("n"):
            new_idx = next_block(components_list, current_idx)
            if new_idx != current_idx:
                current_idx = new_idx
                armed_overwrite = None
                if current_idx >= len(components_list):
                    message = "Fim dos blocos. R=recapturar  Q=sair"
                else:
                    message = f"Pulou para bloco {components_list[current_idx]['block_label']}"
                msg_set_at = time.time()
            continue

        if key == ord("d"):
            if 0 <= current_idx < len(components_list):
                sugg = suggest_digit(components_list[current_idx]["mask_crop"])
                if sugg is not None and delete_template(sugg):
                    saved.pop(sugg, None)
                    message = f"Deletado: '{sugg}.png'"
                else:
                    message = "Nada a deletar (sem sugestao ou arquivo)"
                msg_set_at = time.time()
                armed_overwrite = None
            continue

        if ord("0") <= key <= ord("9") and current_idx < len(components_list):
            digit = chr(key)
            already = (TEMPLATES_DIR / f"{digit}.png").exists()

            if already and armed_overwrite != digit:
                armed_overwrite = digit
                message = ""
                continue

            crop = components_list[current_idx]["mask_crop"]
            path, w_norm = save_template(digit, crop)
            saved[digit] = True
            armed_overwrite = None
            message = f"Salvo: '{digit}' -> {path.name} ({w_norm}x{DIGIT_H})"
            msg_set_at = time.time()
            current_idx += 1
            if current_idx >= len(components_list):
                message += " | Todos processados"
            continue

        armed_overwrite = None

    cv2.destroyAllWindows()
    print("\nResumo dos templates:")
    for d in range(10):
        status = "OK" if saved.get(str(d)) else "FALTA"
        print(f"  {d}: {status}")
    print(f"\nDiretorio: {TEMPLATES_DIR}")


if __name__ == "__main__":
    main()
