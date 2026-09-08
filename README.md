# Snake AI

Un agente que aprende a jugar al Snake solo, por refuerzo.

Comenzó con la idea de crear una IA simple para un juego conocido, en un principio programé el juego con
pygame orientado a una persona sin embargo, este enfoque no resultaba util
a la hora de una IA. Lo primero fue desarmar el juego — sacarle el
timer, separar la lógica del dibujo, convertirlo en algo que avance cuando se
lo piden y en tiempos definidos. Recién ahí se puede entrenar: el juego sin dibujar
corre a unos 117.000 pasos por segundo.

El agente no sabe nada del Snake. No sabe que la fruta es buena ni que las
paredes matan. Solo recibe +10 o -10 después de sus propias decisiones, y de
ahí sale todo lo demás.

## Stack

- **Python 3.13**
- **PyTorch** — la red y el entrenamiento
- **NumPy** — los vectores de estado
- **Matplotlib** — la curva de aprendizaje en vivo
- **Pygame** — el dibujo del juego (opcional: entrenar no lo necesita)

## Cómo funciona

**Lo que ve el agente** son 11 valores 0/1: si hay peligro recto, a la derecha
o a la izquierda; en qué dirección va; y de qué lado quedó la fruta.

**Lo que puede hacer** son 3 acciones relativas: seguir recto, girar a la
derecha, girar a la izquierda. Al ser relativas es imposible pedir un giro de
180° ilegal.

**Lo que recibe** es +10 por comer, -10 por morir, 0 por moverse.

**Cómo aprende**: una red de 11 → 256 → 3 estima qué tan buena es cada acción.
Cada paso corrige esa estimación con la ecuación de Bellman, entrenando dos
veces: sobre el paso que acaba de dar, y sobre un lote de 1.000 recuerdos
sacados al azar de una memoria de 100.000 jugadas pasadas. Esa memoria es la
que evita que se olvide de lo aprendido hace mil partidas.

Al principio juega casi todo al azar. La exploración baja partida a partida
hasta desaparecer alrededor de la número 80.

## Archivos

| Archivo | Qué hace |
|---|---|
| `game.py` | El juego como entorno: `reset()` y `step(action)`. El dibujo es una clase aparte y opcional. |
| `model.py` | La red (`Linear_QNet`) y el paso de entrenamiento (`QTrainer`). |
| `agent.py` | El agente y el bucle de entrenamiento. Guarda checkpoints cada 25 partidas. |
| `helper.py` | El gráfico de score en vivo. |
| `play.py` | Carga un modelo entrenado y lo mira jugar. No entrena ni explora. |
| `model/model_bueno.pth` | Un modelo ya entrenado, listo para usar. |

## Instalación

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install numpy matplotlib pygame
```

La versión CPU de PyTorch es a propósito: para una red de este tamaño la CPU
es más rápida que la GPU, y la descarga pasa de ~2,5 GB a ~200 MB.

## Ver jugar al modelo entrenado

```bash
python play.py --model model_bueno.pth
```

Más rápido, o más partidas:

```bash
python play.py --model model_bueno.pth --fps 30 --games 20
```

Para medirlo sin ventana (mucho más rápido):

```bash
python play.py --model model_bueno.pth --games 200 --no-render
```

## Entrenar

Desde cero, sin dibujar:

```bash
python agent.py --render-every 0
```

Seguir desde el último checkpoint:

```bash
python agent.py --resume --render-every 0
```

Mirando una partida cada 25, para ver cómo va aprendiendo:

```bash
python agent.py --render-every 25
```

Se corta con Ctrl+C. Como el checkpoint se guarda cada 25 partidas, se pierden
como mucho esas.

Cada récord nuevo se guarda en `model/model.pth`, que se sobreescribe en cada
entrenamiento. Cuando aparece un modelo que vale la pena, se copia a mano:

```bash
cp model/model.pth model/model_bueno.pth
```

## Resultados

`model_bueno.pth` salió de 200 partidas de entrenamiento. Medido sobre 200
partidas jugando en serio, sin explorar:

```
200 games | mean 26.57 | best 53 | worst 3
```

## Por qué se estanca cerca de 26

Es una limitación de diseño, no un bug. El agente solo ve **peligro
inmediato**: si la celda de al lado lo mata. No ve su propio cuerpo entero.
Entonces cuando la serpiente ya mide 30 bloques, entra a un hueco que se
cierra detrás de ella y muere ahí. Desde su punto de vista no había ningún
peligro hasta el último paso.

Es como jugar mirando solo las 3 celdas que te rodean: podés esquivar todo lo
que ves venir, pero no podés evitar meterte en un callejón sin salida.

Lo próximo sería darle más contexto — cuántas celdas libres quedan alcanzables
en cada dirección, por ejemplo — o pasar a una red convolucional que vea el
tablero entero.
