import pyautogui
import time

print("Posicione o mouse onde deseja. O programa vai ler em 5 segundos...")
print("1...")
time.sleep(1)
print("Lendo posição do mouse...")

# Pega a posição atual
for i in range(8):
    print(f"Leitura em {i}: 1 segundos...")
    time.sleep(5)

    x, y = pyautogui.position()
    print(f"A posição atual do seu mouse é: X={x}, Y={y}")
