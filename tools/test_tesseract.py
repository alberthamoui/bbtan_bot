"""
testTesseract.py — Verifica se o Tesseract está instalado e funcionando.
Uso: python tools/testTesseract.py
"""

import os
import sys
import subprocess

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("=" * 50)
print("  Teste de Instalação do Tesseract")
print("=" * 50)

# 1. Binário no caminho esperado?
print(f"\n[1] Binário em '{TESSERACT_PATH}':")
if os.path.isfile(TESSERACT_PATH):
    print("    OK — arquivo encontrado")
else:
    print("    FALHOU — arquivo NAO encontrado")
    print("    -> Baixe o instalador em: https://github.com/UB-Mannheim/tesseract/wiki")
    print("    -> Instale e rode este script novamente")

# 2. Tesseract no PATH do sistema?
print("\n[2] Tesseract no PATH do sistema:")
try:
    result = subprocess.run(
        ["tesseract", "--version"], capture_output=True, text=True, timeout=5
    )
    print(f"    OK — {result.stdout.splitlines()[0]}")
except FileNotFoundError:
    print("    FALHOU — nao está no PATH")
    print("    -> Adicione o diretório do Tesseract ao PATH do Windows, ou")
    print("    -> Use o caminho absoluto configurado em config.py")
except Exception as e:
    print(f"    ERRO: {e}")

# 3. pytesseract importável?
print("\n[3] Import pytesseract:")
try:
    import pytesseract

    print("    OK — pytesseract importado")

    # Aponta pro caminho absoluto caso não esteja no PATH
    if os.path.isfile(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    version = pytesseract.get_tesseract_version()
    print(f"    OK — versão detectada: {version}")
except ImportError:
    print("    FALHOU — instale com: pip install pytesseract")
    sys.exit(1)
except pytesseract.TesseractNotFoundError:
    print("    FALHOU — pytesseract não consegue encontrar o binário")
    print(f"    -> Verifique se o arquivo existe: {TESSERACT_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"    ERRO: {e}")
    sys.exit(1)

# 4. OCR funcional com imagem sintética?
print("\n[4] OCR com imagem de teste (número '42'):")
try:
    import numpy as np
    import cv2

    # Cria imagem sintética: fundo branco, texto preto "42"
    img = np.ones((60, 120), dtype=np.uint8) * 255
    cv2.putText(img, "42", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img, config=config).strip()
    if text == "42":
        print(f"    OK — leu '{text}' corretamente")
    else:
        print(f"    ATENCAO — esperava '42', leu '{text}'")
        print("    -> OCR funciona mas pode precisar de ajuste nos parâmetros")
except Exception as e:
    print(f"    ERRO: {e}")

print("\n" + "=" * 50)
