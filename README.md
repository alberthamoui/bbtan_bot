# BBTAN Bot

Bot automatizado para BBTAN rodando em emulador Android no PC.

## Estrutura

```
bbtan_bot/
  config.py       Constantes, cores HSV, regioes, pesos de scoring
  vision.py       Captura de tela + deteccao de blocos/bola (OpenCV)
  physics.py      Motor de simulacao de trajetoria com reflexoes
  executor.py     Conversao angulo -> acao no mouse (PyAutoGUI)
  debug_view.py   Visualizacao da trajetoria sobre o frame
  calibrate.py    Utilitario para ajustar regiao e cores
  main.py         Loop principal do bot
```

## Setup

```bash
pip install -r requirements.txt
```

## Calibracao (OBRIGATORIO antes de rodar)

### 1. Definir a regiao do jogo

```bash
python calibrate.py region
```

Selecione a area do jogo no emulador. Copie o output para `config.py`.

### 2. Ajustar filtros de cor (se necessario)

```bash
python calibrate.py colors
```

Use os sliders HSV ate os blocos ficarem bem isolados na mascara.

### 3. Testar deteccao

```bash
python calibrate.py test
```

Verifique se blocos, power-ups e bola sao detectados corretamente.

## Uso

```bash
# Rodar o bot
python main.py

# Com visualizacao de debug
python main.py --debug

# Simular sem clicar (testa apenas a visao + simulacao)
python main.py --dry-run --debug

# Espera inteligente (detecta fim do turno por visao)
python main.py --smart-wait

# Limitar turnos
python main.py --max-turns 20
```

## Ajustes importantes

Em `config.py`:

- `GAME_REGION`: coordenadas da janela do jogo na tela
- `BLOCK_HSV_LOW/HIGH`: range HSV dos blocos (use calibrate.py colors)
- `TURN_WAIT_TIME`: tempo de espera entre turnos
- `ANGLE_STEP`: resolucao da busca (1 = preciso, 2-5 = mais rapido)
- `execute_shot`: troque entre `execute_shot_tap`, `execute_shot_swipe`,
  ou `execute_shot_hold_release` em `executor.py` conforme seu emulador

## Seguranca

- PyAutoGUI FAILSAFE ativo: mova o mouse para o canto superior esquerdo para abortar
- Ctrl+C para parar o bot a qualquer momento