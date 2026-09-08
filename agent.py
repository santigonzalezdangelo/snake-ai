import torch
import random
import numpy as np
from game import SnakeGame, SnakeRenderer
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
import os
import argparse


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

STRAIGHT = 0
RIGHT = 1
LEFT = 2

RENDER_EVERY = 1
RENDER_FPS = 40

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) #popleft

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return np.array(game.get_state(), dtype=int)

        

    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

        #popleft if max memory

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, next_states, rewards, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, next_states, rewards, dones)

    def train_short_memory(self, state, action, next_state, reward, done):
        self.trainer.train_step(state, action, next_state, reward, done)


    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            prediction = self.model(torch.tensor(state, dtype=torch.float))
        return torch.argmax(prediction).item()
    #para el video 4 target[idx][action[idx].item()] = Q_new

    def save_checkpoint(self, path="./model/checkpoint.pth", record=0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "n_games": self.n_games,
            "record": record,
        }, path)

    def load_checkpoint(self, path="./model/checkpoint.pth"):
        if not os.path.exists(path):
            return None
        ck = torch.load(path)
        self.model.load_state_dict(ck["model"])
        self.trainer.optimizer.load_state_dict(ck["optimizer"])
        self.n_games = ck["n_games"]
        return ck["record"]



def train(render_every=RENDER_EVERY, resume=False, save_every=25):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()

        if resume:
            loaded = agent.load_checkpoint()
            if loaded is None:
                print("No checkpoints")
            else:
                record = loaded
                print("Starting from checkpoint")

        game = SnakeGame()
        renderer = SnakeRenderer(game, "Snake AI - Training") if render_every else None
        while True:
            show = render_every and (agent.n_games + 1) % render_every == 0

            state_old = agent.get_state(game)

            final_move = agent.get_action(state_old)

            state_new, reward, done = game.step(final_move)
            score = game.score 

            #short memory training
            agent.train_short_memory(state_old, final_move, state_new, reward, done)

            agent.remember(state_old, final_move, state_new, reward, done)

            if show and renderer and not renderer.draw(fps=RENDER_FPS):
                renderer.close(); renderer = None; render_every = 0

            if done:
                #long memory training and plot results
                game.reset()
                agent.n_games+=1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                
                print("Game: ", agent.n_games, "Score: ", score, "Record: ", record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                if agent.n_games % 5 == 0:
                    plot(plot_scores, plot_mean_scores)
                if agent.n_games % save_every == 0:
                    agent.save_checkpoint(record=record)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", help="Starting from checkpoint")
    p.add_argument("--render-every", type=int, default=RENDER_EVERY)
    args = p.parse_args()
    train(render_every=args.render_every, resume=args.resume)

