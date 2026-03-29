"""
This file includes the functions related to the epsilon-greedy policy. 
It also includes the n_actions variable, which is set to 4 for the gymnasium environment. 
"""

import torch, random

n_actions = 6


def epsilon(iteration, initial_epsilon, min_epsilon, min_epsilon_iteration):
    """
    A relation between the iteration number and epsilon-value;
    the relation is a linearly decreasing function, starting at initial_epsilon 
    and ending in a constant value min_epsilon from min_epsilon_iteration. 
    """
    return max(min_epsilon, initial_epsilon - (initial_epsilon-min_epsilon)*iteration/min_epsilon_iteration)


def get_action(Q, epsilon, verbose=False):
    """
    This function uses an epsilon-greedy policy to determine the next action. 
    Output is a one-hot encoding. 
    """
    global n_actions

    random_number = random.random()
    action = torch.zeros(n_actions)

    if Q is None or random_number < epsilon: 
        # take a random action with probability epsilon
        action[random.randint(0, n_actions-1)] = 1
    else: 
        # take the greedy action with probability (1 - epsilon)
        action[torch.argmax(Q)] = 1

        if verbose: 
            if torch.argmax(Q) == 3:
                print("Model chose left")
            elif torch.argmax(Q) == 2:
                print("Model chose right")
            else:
                print("Model chose to do nothing")
    return action
