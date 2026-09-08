import argparse, torch, numpy as np
from game import SnakeGame, SnakeRenderer
from model import Linear_QNet

def play(games=10, fps=12, render=True, file_name="model.pth"):
    model = Linear_QNet(11, 256, 3)
    if not model.load(file_name):
        print("Train first")
        return
    model.eval()
    game = SnakeGame()
    renderer = SnakeRenderer(game, "Snake AI - Playing") if render else None
    scores = []
    for i in range(1, games + 1):
        state = game.reset()
        while True:
            with torch.no_grad():
                move = torch.argmax(model(torch.tensor(np.array(state), dtype=torch.float))).item()
            state, _, done = game.step(move)
            if renderer and not renderer.draw(fps=fps):
                renderer.close(); renderer = None
            if done:
                break
        scores.append(game.score)
        print("Game {}: Score {}".format(i, game.score))
        if renderer is None and render:
            break

    if renderer:
        renderer.close()
    if scores:
        print("\n{} games | mean {:.2f} | best {} | worst {}".format(
            len(scores), sum(scores)/len(scores), max(scores), min(scores)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--model", default="model.pth")
    a = p.parse_args()
    play(games=a.games, fps=a.fps, render=not a.no_render, file_name=a.model)