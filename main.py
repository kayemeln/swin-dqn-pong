import gymnasium as gym
import ale_py
import torch, random, os, copy, csv
from collections import deque
from functools import partial
import models, states, actions, plotting
import numpy as np


# ----- SETTINGS ----- #

# interface & runtime
name = "CNN_Tennis_NoFrameskip"
load_model = None  # set to a .pth path to resume training, e.g. 'results/CNN_2/CNN_2_100000.pth'
render = False
n_iterations = int(1e7)
save_every = int(1e5)

# preprocessing
states.img_size = (84, 84)
states.n_frames = 4
actions.n_actions = 18
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
replay_start_size = 80000        # fill replay buffer before training
target_update_freq = 1000        # how often to sync target network
eval_every = 10                  # run evaluation every N training episodes
eval_episodes = 2                # number of greedy evaluation episodes
update_frequency = 4

if load_model:
    model = torch.load(load_model, map_location='cpu', weights_only=False)
    print(f"Loaded model from {load_model}")
else:
#    model = models.VisionTransformer(img_size=states.img_size[0], n_frames=states.n_frames, num_actions=actions.n_actions)
#    model = models.SwinDQN(states.n_frames, actions.n_actions)
    model = models.ConvModel(states.img_size, states.n_frames, actions.n_actions)
target_model = copy.deepcopy(model)
target_model.eval()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = torch.nn.MSELoss()


# ----- INITIALIZATION ----- #

# initialize the environment
env = gym.make("TennisNoFrameskip-v4", render_mode=('human' if render else None))
env = states.modify_gym_env(env)  # activate the preprocessing functions
env.metadata['render_fps'] = 60  # only used if render=True

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
target_model.to(device)

# initialize storage
D = deque(maxlen=10**6)
iterations, scores, losses, epsilons = [], [], [], []
eval_iterations, eval_scores, eval_ma_scores = [], [], []
if not os.path.exists('results/'+name): os.makedirs('results/'+name)  # create a folder to store the results

# open CSV log file
csv_file = open('results/'+name+'/'+name+'_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['iteration', 'type', 'score', 'ma_score', 'loss', 'epsilon'])


def run_eval_episodes(model, n_episodes):
    """Run n_episodes with greedy policy (epsilon=0) and return the average score."""
    eval_env = gym.make("TennisNoFrameskip-v4", render_mode=None)
    eval_env = states.modify_gym_env(eval_env)
    total_score = 0
    for _ in range(n_episodes):
        s, _ = eval_env.reset()
        done = False
        ep_score = 0
        while not done:
            with torch.no_grad():
                q = model(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device))
            a = actions.get_action(q, epsilon=0.0)
            s, r, done, truncated, info = eval_env.step(torch.argmax(a).item())
            if truncated: done = True
            ep_score += r
            if done and eval_env.unwrapped.ale.lives() > 0:
                s, _ = eval_env.reset()
                done = False
        total_score += ep_score
    eval_env.close()
    return total_score / n_episodes


# set some variables for the loop
terminated = True
quit = False
save = False
train_step = 0
episode_count = 0

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
                losses += [0]
                epsilons += [epsilon]
                print(f"\rFilling replay buffer... ({len(D)}/{replay_start_size})")
            continue

        train_step += 1

        if i % update_frequency == 0:
            # step 3: create a minibatch for learning
            minibatch = random.sample(D, min(len(D), minibatch_size))
            states_batch, actions_batch, rewards_batch, next_states_batch, terminations_batch = tuple([*zip(*minibatch)])

            states_batch = torch.tensor(np.array(states_batch), dtype=torch.float32).to(device)
            next_states_batch = torch.tensor(np.array(next_states_batch), dtype=torch.float32).to(device)
            actions_batch = torch.stack(actions_batch).to(device)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
            terminations_batch = torch.tensor(terminations_batch, dtype=torch.float32).to(device)

            # step 4: compute learning targets using Double DQN
            with torch.no_grad():
                online_q = model.forward(next_states_batch)
                best_actions = torch.argmax(online_q, dim=1)
                target_q = target_model.forward(next_states_batch)
                y_batch = rewards_batch + discount * target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1) * (1 - terminations_batch)

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
            ep_loss_mean = torch.mean(torch.tensor(ep_loss)).item() if ep_loss else 0
            losses += [ep_loss_mean]
            epsilons += [epsilon]
            episode_count += 1

            csv_writer.writerow([i, 'train', score, '', f"{ep_loss_mean:.6f}", f"{epsilon:.4f}"])

            print(f"\rEnd of game (iteration {i}), score: {score}, epsilon: {epsilon:.3f}")

            # run greedy evaluation episodes periodically
            if episode_count % eval_every == 0:
                model.eval()
                avg_eval_score = run_eval_episodes(model, eval_episodes)
                model.train()
                eval_iterations += [i]
                eval_scores += [avg_eval_score]
                eval_ma_scores += [torch.mean(torch.tensor(eval_scores[-10:])).item()]
                csv_writer.writerow([i, 'eval', avg_eval_score, f"{eval_ma_scores[-1]:.4f}", '', ''])
                csv_file.flush()
                print(f"  [EVAL] avg greedy score: {avg_eval_score:.1f}, eval MA-10: {eval_ma_scores[-1]:.2f}")

        if save:
            plotting.save_plot(iterations, scores, losses, epsilons, name,
                               eval_iterations=eval_iterations, eval_scores=eval_scores, eval_ma_scores=eval_ma_scores)
            torch.save(model, 'results/'+name+'/'+name+'_'+str(i)+'.pth')
            save = False

except KeyboardInterrupt:
    print(f"\nInterrupted at iteration {i}. Saving model...")

# end of training, save the model, the training logs and a plot
plotting.save_plot(iterations, scores, losses, epsilons, name,
                   eval_iterations=eval_iterations, eval_scores=eval_scores, eval_ma_scores=eval_ma_scores)
torch.save(model, 'results/'+name+'/'+name+'.pth')
csv_file.close()
print('Done.')
