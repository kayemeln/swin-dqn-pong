import sys
import time
import torch
import gymnasium as gym
import ale_py
import states, actions


if len(sys.argv) < 2:
    print("Usage: python play.py <model.pth>")
    sys.exit(1)

model_path = sys.argv[1]

# ----- SETTINGS ----- #

states.img_size = (84, 84)
states.n_frames = 4
actions.n_actions = 6

# ----- LOAD MODEL ----- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, weights_only=False, map_location=device)
model.eval()
print(f"Loaded model from {model_path} (device: {device})")

# ----- INITIALIZATION ----- #

env = gym.make("PongNoFrameskip-v4", render_mode='human')
env = states.modify_gym_env(env)
env.metadata['render_fps'] = 60

# ----- GAME LOOP ----- #

state = env.reset()[0]
terminated = False
score = 0

while True:
    with torch.no_grad():
        output = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
    action = actions.get_action(output, epsilon=0.0)
    # print(f"Action: {torch.argmax(action).item()} | Time: {time.time():.4f}")

    next_state, reward, terminated, truncated, info = env.step(torch.argmax(action).item())
    if truncated: terminated = True

    score += reward

    if terminated and (env.unwrapped.ale.lives() == 0):
        break

    if terminated:
        state = env.reset()[0]
        terminated = False
    else:
        state = next_state

print(f"Final score: {score}")
del env
