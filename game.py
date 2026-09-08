import random
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

#direcciones
CLOCKWISE = [Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)]

#acciones
STRAIGHT = 0
RIGHT = 1
LEFT = 2

#recompensas
REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_STEP = 0.0

class SnakeGame:
    def __init__(self, cell_number=20, max_steps_factor=100, seed=None):
        self.cell_number = cell_number
        self.max_steps_factor = max_steps_factor
        self.rng = random.Random(seed)
        self.reset()

    #ciclo principal del juego

    def reset(self):
        mid = self.cell_number // 2
        self.body = [Point(5, mid), Point(4, mid), Point(3, mid)]
        self._body_set = set(self.body)
        self.direction = Point(1, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.won = False
        self._place_food()
        return self.get_state()

    def step(self, action):
        if self.game_over:
            raise RuntimeError("Termino la partida")

        self.steps += 1
        self.steps_since_food += 1
        self.direction = self._turn(self.direction, action)

        head = self.body[0]
        new_head = Point(head.x + self.direction.x, head.y + self.direction.y)

        if self._hits_wall(new_head):
            self.game_over = True
            return self.get_state(), REWARD_DEATH, True

        ate = new_head == self.food
        self.body.insert(0, new_head)
        if not ate:
            tail = self.body.pop()
            if tail not in self.body:
                self._body_set.discard(tail)
        self._body_set.add(new_head)

        if new_head in self.body[1:]:
            self.game_over = True
            return self.get_state(), REWARD_DEATH, True

        if ate:
            self.score += 1
            self.steps_since_food = 0
            if not self._place_food():
                self.won = True
                self.game_over = True
                return self.get_state(), REWARD_FOOD, True
            return self.get_state(), REWARD_FOOD, False

        if self.steps_since_food > self.max_steps_factor * len(self.body):
            self.game_over = True
            return self.get_state(), REWARD_DEATH, True

        return self.get_state(), REWARD_STEP, False


    def get_state(self):

        #11 valores booleanos, peligro relativo, direccion actual y donde esta la fruta

        head = self.body[0]
        d = self.direction
        right = self._turn(d, RIGHT)
        left = self._turn(d, LEFT)

        pt_straight = Point(head.x + d.x, head.y + d.y)
        pt_right = Point(head.x + right.x, head.y + right.y)
        pt_left = Point(head.x + left.x, head.y + left.y)

        return [
            int(self.is_collision(pt_straight)),  # peligro recto
            int(self.is_collision(pt_right)),  # peligro a la derecha
            int(self.is_collision(pt_left)),  # peligro a la izquierda

            int(d == Point(-1, 0)),  # direccion izquierda
            int(d == Point(1, 0)),  # direccion derecha
            int(d == Point(0, -1)),  # direccion arriba
            int(d == Point(0, 1)),  # direccion abajo

            int(self.food.x < head.x),  # comida a la izquierda
            int(self.food.x > head.x),  # comida a la derecha
            int(self.food.y < head.y),  # comida arriba
            int(self.food.y > head.y)  # comida abajo
        ]

    def get_grid(self):
        #Vista alternativa: la grilla entera (0 vacio, 1 cuerpo, 2 cabeza, 3 fruta)
        grid = [[0] * self.cell_number for _ in range(self.cell_number)]
        for block in self.body[1:]:
            grid[block.y][block.x] = 1
        head = self.body[0]
        if not self._hits_wall(head):
            grid[head.y][head.x] = 2
        grid[self.food.y][self.food.x] = 3
        return grid

    def is_collision(self, pt):
        if self._hits_wall(pt):
            return True
        return pt in self._body_set and pt != self.body[-1]

    def _hits_wall(self, pt):
        return not (0 <= pt.x < self.cell_number and 0 <= pt.y < self.cell_number)

    def _turn(self, direction, action):
        i = CLOCKWISE.index(direction)
        if action == RIGHT:
            i = (i+1)%4
        elif action == LEFT:
            i = (i-1)%4
        return CLOCKWISE[i]

    def _place_food(self):
        free =[
            Point(x, y) 
            for x in range(self.cell_number) 
            for y in range(self.cell_number)
            if Point(x, y) not in self._body_set
        ]
        if not free:
            return False
        self.food = self.rng.choice(free)
        return True

class SnakeRenderer:
    CELL_SIZE = 40
    BG = (175, 215, 70)
    GRASS = (167, 209, 61)
    SNAKE = (40, 120, 100)
    HEAD = (30, 90, 75)
    FRUIT = (220, 30, 30)
    TEXT = (56, 74, 12)

    def __init__(self, game, caption="Snake Game"):
        import pygame
        self.pygame = pygame
        self.game = game
        pygame.init()
        size = game.cell_number * self.CELL_SIZE
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)

    def draw(self, fps=10):
        pygame=self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(self.BG)
        self._draw_grass()

        g = self.game
        cs = self.CELL_SIZE
        pygame.draw.rect(self.screen, self.FRUIT,
                         pygame.Rect(g.food.x * cs, g.food.y * cs, cs, cs))
        for i, block in enumerate(g.body):
            color = self.HEAD if i == 0 else self.SNAKE
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(block.x * cs, block.y * cs, cs, cs))

        surf = self.font.render(str(g.score), True, self.TEXT)
        end = g.cell_number * cs
        self.screen.blit(surf, surf.get_rect(center=(end - 60, end - 40)))

        pygame.display.update()
        self.clock.tick(fps)
        return True

    def _draw_grass(self):
        cs = self.CELL_SIZE
        for row in range(self.game.cell_number):
            for col in range(self.game.cell_number):
                if (row + col) % 2 == 0:
                    self.pygame.draw.rect(
                        self.screen, self.GRASS,
                        self.pygame.Rect(col * cs, row * cs, cs, cs))

    def close(self):
        self.pygame.quit()



def play_human():
    import pygame
    game = SnakeGame()
    r = SnakeRenderer(game, "Snake - humano")
    pending = STRAIGHT
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Traduce la tecla absoluta a la accion relativa equivalente.
                wanted = {pygame.K_RIGHT: Point(1, 0), pygame.K_LEFT: Point(-1, 0),
                          pygame.K_UP: Point(0, -1), pygame.K_DOWN: Point(0, 1)}.get(event.key)
                if wanted:
                    for a in (STRAIGHT, RIGHT, LEFT):
                        if game._turn(game.direction, a) == wanted:
                            pending = a
                            break
        _, _, done = game.step(pending)
        pending = STRAIGHT
        if not r.draw(fps=8):
            running = False
        if done:
            print("Score:", game.score)
            game.reset()
    r.close()

def play_random():
    game = SnakeGame()
    r = SnakeRenderer(game, "Snake - random")
    while True:
        _, _, done = game.step(random.choice([STRAIGHT, RIGHT, LEFT]))
        if not r.draw(fps=15):
            break
        if done:
            print("Score:", game.score, "| pasos:", game.steps)
            game.reset()
    r.close()

def bench(n=200000):
    import time
    game = SnakeGame(seed=0)
    rng = random.Random(0)
    t0 = time.perf_counter()
    for _ in range(n):
        _, _, done = game.step(rng.choice([STRAIGHT, RIGHT, LEFT]))
        if done:
            game.reset()
    dt = time.perf_counter() - t0
    print("{} pasos en {:.2f}s -> {:,.0f} pasos/seg".format(n, dt, n / dt))

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "human"
    {"human": play_human, "random": play_random, "bench": bench}[mode]()

