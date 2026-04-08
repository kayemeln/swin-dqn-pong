import sys
import time
import torch
import gymnasium as gym
import ale_py
import states, actions

# Check command line arguments
# Expect a model file path as input
if len(sys.argv) < 2:
    print("Usage: python play.py <model.pth>")
    sys.exit(1)

model_path = sys.argv[1]

# Define input dimensions and action space
states.img_size = (84, 84)
states.n_frames = 4
actions.n_actions = 6

# Load trained model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, weights_only=False, map_location=device)
model.eval()
print(f"Loaded model from {model_path} (device: {device})")

# Environment initialisation 
env = gym.make("PongNoFrameskip-v4", render_mode='human')
env = states.modify_gym_env(env)
env.metadata['render_fps'] = 60

# Game loop
# Reset environment and get initial state
state = env.reset()[0]
terminated = False
score = 0

while True:
    with torch.no_grad():
        output = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
    # select action using greedy policy
    action = actions.get_action(output, epsilon=0.0)
    # print(f"Action: {torch.argmax(action).item()} | Time: {time.time():.4f}")

    # step environment using selected action 
    next_state, reward, terminated, truncated, info = env.step(torch.argmax(action).item())
    if truncated: terminated = True

    # accumulate score
    score += reward

    # end game when all lives are lost
    if terminated and (env.unwrapped.ale.lives() == 0):
        break

    # if episode ends but lives remain, reset environment
    if terminated:
        state = env.reset()[0]
        terminated = False
    else:
        state = next_state

# Final output
print(f"Final score: {score}")
del env
