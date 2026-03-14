"""
This file is used to train a model according to the Reinforcement Learning loop.
Hyperparameters can be chosen under 'settings', and some code to inspect the states is included as comments.
This code will periodically save the model along with some performance metrics, so different stages can later be compared.

'demo.py' can be used to see the model playing the game.
For faster training, 'quick_train.py' can be used.
"""

import gymnasium as gym
import ale_py
import torch, random, os, copy
from collections import deque
from functools import partial
import models, states, actions, plotting
import numpy as np


# ----- SETTINGS ----- #

# interface & runtime
name = "CNN_3"
load_model = "None"  # set to a .pth path to resume training, e.g. 'results/CNN_2/CNN_2_100000.pth'
render = False
n_iterations = int(1e6)
save_every = int(1e5)

# preprocessing
states.img_size = (84, 84)
states.n_frames = 4
actions.n_actions = 6
minibatch_size = 32

# randomness
epsilon_fn = partial(actions.epsilon,
    initial_epsilon=0.01 if load_model else 1,
    min_epsilon=0.01,
    min_epsilon_iteration=0.1*n_iterations  # 100k steps of exploration
    )

# training & model
learning_rate = 1e-4
discount = 0.99
replay_start_size = 10000        # fill replay buffer before training
target_update_freq = 1000        # how often to sync target network

if load_model:
    model = torch.load(load_model, map_location='cpu', weights_only=False)
    print(f"Loaded model from {load_model}")
else:
    model = models.ConvModel(states.img_size, states.n_frames, actions.n_actions)
target_model = copy.deepcopy(model)
target_model.eval()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = torch.nn.MSELoss()


# ----- INITIALIZATION ----- #

# initialize the environment
env = gym.make("ALE/Pong-v5", render_mode=('human' if render else None))
env = states.modify_gym_env(env)  # activate the preprocessing functions
env.metadata['render_fps'] = 60  # only used if render=True

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
target_model.to(device)

# initialize storage
D = deque(maxlen=10**5)
iterations, scores, ma_scores, losses, av_Qs = [], [], [], [], []
if not os.path.exists('results/'+name): os.makedirs('results/'+name)  # create a folder to store the results

# set some variables for the loop
terminated = True
quit = False
save = False
train_step = 0


# ----- TRAINING LOOP ----- #

try:
    for i in range(n_iterations):
        if terminated:
            # reset the game
            state, _ = env.reset()
            epsilon = epsilon_fn(train_step)
            reward, score, ep_loss = 0, 0, []
            terminated = False

        # step 1: take action according to epsilon-greedy policy
        output = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        action = actions.get_action(output, epsilon)

        next_state, reward, terminated, truncated, info = env.step(torch.argmax(action).item())
        if truncated: terminated = True
        score += reward

        # step 2: add the new variables to the memory
        D.append((state, action, reward, next_state, terminated))

        # only start training after replay buffer has enough samples
        if len(D) < replay_start_size:
            state = next_state
            if terminated:
                iterations += [i]
                scores += [score]
                ma_scores += [torch.mean(torch.tensor(scores[-100:])).item()]
                losses += [0]
                print(f"\rFilling replay buffer... ({len(D)}/{replay_start_size})")
            continue

        train_step += 1

        # step 3: create a minibatch for learning
        minibatch = random.sample(D, min(len(D), minibatch_size))
        states_batch, actions_batch, rewards_batch, next_states_batch, terminations_batch = tuple([*zip(*minibatch)])

        states_batch = torch.tensor(np.array(states_batch), dtype=torch.float32).to(device)
        next_states_batch = torch.tensor(np.array(next_states_batch), dtype=torch.float32).to(device)
        actions_batch = torch.stack(actions_batch).to(device)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        terminations_batch = torch.tensor(terminations_batch, dtype=torch.float32).to(device)

        # step 4: compute learning targets using the TARGET network
        with torch.no_grad():
            target_q = target_model.forward(next_states_batch)
            y_batch = rewards_batch + discount * torch.max(target_q, dim=1).values * (1 - terminations_batch)

        # step 5: optimize
        optimizer.zero_grad()
        y_pred = torch.sum(model.forward(states_batch) * actions_batch, dim=1)
        loss = loss_fn(y_pred, y_batch)
        if torch.isnan(loss):
            print("Training failed")
            quit = True
            break
        loss.backward()
        optimizer.step()
        ep_loss += [loss.item()]

        # periodically update the target network
        if i % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # end of iteration
        state = next_state
        if i % save_every == 0: save = True  # periodically save some variables

        if terminated:
            # print/plot/save some important metrics
            iterations += [i]
            scores += [score]
            ma_scores += [torch.mean(torch.tensor(scores[-100:])).item()]
            losses += [torch.mean(torch.tensor(ep_loss)).item() if ep_loss else 0]

            print(f"\rEnd of game (iteration {i}), episode_score:{score:.2f}, ma_score: {ma_scores[-1]:.2f}, epsilon: {epsilon:.3f}")

        if save:
            plotting.save_plot(iterations, ma_scores, name)
            torch.save(model, 'results/'+name+'/'+name+'_'+str(i)+'.pth')
            save = False

except KeyboardInterrupt:
    print(f"\nInterrupted at iteration {i}. Saving model...")

# end of training, save the model, the training logs and a plot
plotting.save_plot(iterations, ma_scores, name)
#plotting.save_data(iterations, scores, av_Qs, losses, name)
torch.save(model, 'results/'+name+'/'+name+'.pth')
print('Done.')
