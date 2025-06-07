import os
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt

dx = 3/np.sqrt(3)

class CyclicCounter:
    def __init__(self, n, start=0):
        self.n = n
        self.current = start % n
    def inc(self):
        self.current = (self.current + 1) % self.n
    def get(self):
        return self.current

def dist(pos_1, pos_2):
    return np.sqrt(((pos_1[0]-pos_2[0]) * dx)**2 + (2*(pos_1[1]-pos_2[1]))**2)

def xy2ij_obs(x, y):
    return int((y-0.5)*2)-1, int(x)
def ij2xy_obs(i, j):
    if (i+j) % 2 == 0:
        if j > 0:
            j -= 1
        else:
            j += 1
    return int(j), (i+1)/2+0.5

def evaluate_rewards(game, policy, n_games=1, t_max=1000, device='cuda'):
    rewards_1 = []
    rewards_2 = []
    winners = []
    for _ in range(n_games):
        s = game.reset()
        R = [0, 0]
        t = 0
        mem_1, mem_2 = torch.zeros(1, 256, dtype=torch.float32, device=device), torch.zeros(1, 256, dtype=torch.float32, device=device)
        for _ in range(t_max):
            act = policy[game.team_counter.get()].act(np.array([s]), mem_1=mem_1, mem_2=mem_2)
            dist, actions = act['dist'], act['actions']
            mem_1, mem_2 = act['memory']
            r, s, done = game.step(dist.decode_action(actions.argmax()))
            R[t] += r
            if done:
                winners.append(t)
                break
            t = game.team_counter.get()
        rewards_1.append(R[0])
        rewards_2.append(R[1])
    eval_res = {
                    'r_1' : np.array(rewards_1), 
                    'r_2' : np.array(rewards_2), 
                    'winners' : np.array(winners)}
    return eval_res

def plot_learning_stats(eval_results, plot_both=False):
    r_1 = np.array([eval['r_1'] for eval in eval_results]).flatten()
    r_2 = np.array([eval['r_2'] for eval in eval_results]).flatten()
    r_1_mean = np.array([eval['r_1'].mean() for eval in eval_results])
    r_2_mean = np.array([eval['r_2'].mean() for eval in eval_results])
    all_r = pd.DataFrame({
        'r_1' : r_1, 'r_2' : r_2, 
        'x' : [i//len(eval_results[0]['r_1']) for i in range(len(r_1))]
    })

    schema = '''
        AAABB
    '''
    # Clear current figure
    plt.clf()
    clear_output(True)
    fig, axs = plt.subplot_mosaic(schema)
    
    plot_rewards(r_1, all_r, ax=axs['A'])
    plot_compare(r_1_mean, r_2_mean, all_r, ax=axs['B'])

    plt.draw()  # Force redraw
    plt.pause(0.1)  # Process GUI events
    plt.close()  # Close figure to prevent memory leaks

def plot_rewards(reward, all_r, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(reward, label='Average Reward', color='blue')
    ax.scatter(all_r['x'], all_r['r_1'], color='red')
    ax.xlabel('iter')
    ax.ylabel('reward')
    ax.set_title('Player №1 rewards')
    ax.legend()
    ax.grid(True)

def plot_compare(r_1, r_2, all_r=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    d = r_1 - r_2
    ax.plot(d)
    if all_r is not None:
        ax.scatter(all_r['x'], all_r['r_1'] - all_r['r_2'], color='red')

def load_weights(model, file_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.load_state_dict(file_path, map_location=device)