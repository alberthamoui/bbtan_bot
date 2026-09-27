"""
consolidate_hp_templates.py — Consolida amostras de digitos HP em templates medios.

Para cada digito 0-9, le todas as imagens templates/hp_digits/<digito>_*.png,
redimensiona pra altura 20px mantendo aspect ratio (usa a largura mediana)
e tira a media dos pixels. Salva o resultado como templates/hp_digits/<digito>.png.
"""

from pathlib import Path
import cv2
import numpy as np

HP_DIGIT_H = 20
DIGITS_DIR = Path(__file__).resolve().parent.parent / "templates" / "hp_digits"


def consolidate_digit(digit: str) -> bool:
    samples = sorted(DIGITS_DIR.glob(f"{digit}_*.png"))
    if not samples:
        print(f"  [{digit}] sem amostras — pulando")
        return False

    resized_imgs = []
    widths = []
    for p in samples:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        src_h, src_w = img.shape[:2]
        aspect = src_w / max(src_h, 1)
        new_w = max(1, round(HP_DIGIT_H * aspect))
        resized = cv2.resize(img, (new_w, HP_DIGIT_H), interpolation=cv2.INTER_AREA)
        resized_imgs.append(resized)
        widths.append(new_w)

    if not resized_imgs:
        print(f"  [{digit}] amostras invalidas — pulando")
        return False

    target_w = int(np.median(widths))
    target_w = max(1, target_w)

    normalized = []
    for img in resized_imgs:
        if img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, HP_DIGIT_H), interpolation=cv2.INTER_AREA)
        normalized.append(img.astype(np.float32))

    stack = np.stack(normalized, axis=0)
    mean_img = np.mean(stack, axis=0)
    _, binary = cv2.threshold(mean_img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    out_path = DIGITS_DIR / f"{digit}.png"
    cv2.imwrite(str(out_path), binary)
    print(
        f"  [{digit}] {len(samples)} amostras -> {out_path.name} ({target_w}x{HP_DIGIT_H})"
    )
    return True


def main():
    if not DIGITS_DIR.exists():
        print(f"Pasta nao existe: {DIGITS_DIR}")
        return
    print(f"Consolidando templates em {DIGITS_DIR}")
    for d in "0123456789":
        consolidate_digit(d)
    print("Pronto.")


if __name__ == "__main__":
    main()
