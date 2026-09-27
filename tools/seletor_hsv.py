import cv2
import numpy as np
import mss
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg


def nada(x):
    pass


# Criar uma janela para os controles
cv2.namedWindow("Seletor HSV: Controles")
cv2.resizeWindow("Seletor HSV: Controles", 400, 300)

# Criar barras de ajuste (Trackbars) para o limite Inferior (LOW)
cv2.createTrackbar("H_MIN", "Seletor HSV: Controles", 0, 179, nada)
cv2.createTrackbar("S_MIN", "Seletor HSV: Controles", 0, 255, nada)
cv2.createTrackbar("V_MIN", "Seletor HSV: Controles", 200, 255, nada)

# Criar barras de ajuste (Trackbars) para o limite Superior (HIGH)
cv2.createTrackbar("H_MAX", "Seletor HSV: Controles", 179, 179, nada)
cv2.createTrackbar("S_MAX", "Seletor HSV: Controles", 41, 255, nada)
cv2.createTrackbar("V_MAX", "Seletor HSV: Controles", 255, 255, nada)

print("=== CALIBRADOR HSV ATIVO ===")
print("Instruções: Ajuste as barras até que APENAS o objeto desejado fique BRANCO")
print("Pressione 'q' na janela do jogo para sair e salvar os valores.")

sct = mss.MSS()

while True:
    # Captura a tela usando a região definida no seu config.py
    monitor = {
        "top": 244,
        "left": 1351,
        "width": 461,
        "height": 672,
    }

    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Converte a imagem capturada para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ler os valores atuais das Trackbars
    h_min = cv2.getTrackbarPos("H_MIN", "Seletor HSV: Controles")
    s_min = cv2.getTrackbarPos("S_MIN", "Seletor HSV: Controles")
    v_min = cv2.getTrackbarPos("V_MIN", "Seletor HSV: Controles")

    h_max = cv2.getTrackbarPos("H_MAX", "Seletor HSV: Controles")
    s_max = cv2.getTrackbarPos("S_MAX", "Seletor HSV: Controles")
    v_max = cv2.getTrackbarPos("V_MAX", "Seletor HSV: Controles")

    # Define os arrays no formato que o OpenCV e seu config usam
    hsv_low = np.array([h_min, s_min, v_min])
    hsv_high = np.array([h_max, s_max, v_max])

    # Aplica o filtro (Máscara) na imagem
    mascara = cv2.inRange(hsv, hsv_low, hsv_high)

    # Mostra o resultado da filtragem e a imagem original
    cv2.imshow("Seletor HSV: Mask da cor", mascara)
    cv2.imshow("Seletor HSV: Frame do emulador", frame)

    # Se pressionar 'q', interrompe o loop e exibe o resultado formatado
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n=== VALORES EXTRAÍDOS COM SUCESSO! ===")
        print("Copie e cole as linhas abaixo no seu arquivo config.py:\n")
        print(f"OBJETO_HSV_LOW  = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"OBJETO_HSV_HIGH = np.array([{h_max}, {s_max}, {v_max}])")
        break

cv2.destroyAllWindows()
