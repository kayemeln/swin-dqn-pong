"""
This file contains all the code used for plotting and saving images and data.
"""

import matplotlib.pyplot as plt
import numpy as np
import actions


def save_data(iterations, scores, average_Qs, losses, name):
    """
    This function saves a number of important metrics to a .txt file.
    """
    data = np.vstack((np.array(iterations, dtype=float), np.array(scores, dtype=float), np.array(average_Qs, dtype=float), np.array(losses, dtype=float))).T
    np.savetxt('results/'+name+'/'+name+'_data.txt', data, delimiter=';', header='iteration;score;average_Q;loss')


def save_plot(iterations, scores, losses, epsilons, name, i='Done',
              eval_iterations=None, eval_scores=None, eval_ma_scores=None):
    """
    This function saves training summary plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Training Progress')

    axes[0].plot(iterations, scores, alpha=0.3, color='steelblue', label='Train score')
    if eval_iterations and eval_scores:
        axes[0].plot(eval_iterations, eval_scores, alpha=0.3, color='darkorange', label='Eval score')
        axes[0].plot(eval_iterations, eval_ma_scores, color='darkorange', label='Eval MA-10')
    axes[0].set_ylabel('Score')
    axes[0].legend(loc='upper left')

    axes[1].plot(iterations, losses, color='crimson')
    axes[1].set_ylabel('Loss')

    axes[2].plot(iterations, epsilons, color='forestgreen')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel('Iteration')

    plt.tight_layout()
    plt.savefig('results/'+name+'/'+name+'_'+str(i))
    plt.close(fig)


def plot_epsilon(initial_epsilon=1., min_epsilon=0.1, min_epsilon_iteration=10**4):
    """
    This function creates and saves a plot of the value epsilon over the iterations. 
    """

    x = np.arange(10**5)
    y = []
    for x_prime in x:
        y += [actions.epsilon(x_prime, initial_epsilon=initial_epsilon, min_epsilon=min_epsilon, min_epsilon_iteration=min_epsilon_iteration)]
    print(y[0])
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(x, y)
    ax.set_title("$\epsilon$ as a Function of the Iteration Number")
    ax.set_ylabel("$\epsilon$")
    ax.set_xlabel("Iterations")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    plt.savefig('epsilon_function')


def plot_data(col, saveas, path1, label1, path2=None, label2=None, n=None):
    """
    This function creates and saves a plot of the performance of one or two models. 
    """

    cols = ['Reward', 'Q', 'Loss']

    # loading the data
    data1 = np.loadtxt(path1, delimiter=';')
    data2 = []
    if path2 is not None:
        data2 = np.loadtxt(path2, delimiter=';')
    x1, x2 = np.arange(len(data1)), np.arange(len(data2))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if n == 'all':
        av1 = np.cumsum(data1[:, col]) / (x1 + 1)
        ax.plot(data1[:, 0], av1, label=label1)
        if path2 is not None: 
            av2 = np.cumsum(data2[:, col]) / (x2 + 1)
            ax.plot(data2[:, 0], av2, label=label2)
            ax.legend()

    elif n is not None: 
        ma = np.convolve(data1[:, col], np.ones(n)/n, mode='valid')
        ax.plot(data1[(n-1):, 0], ma, label=label1)
        if path2 is not None: 
            ma2 = np.convolve(data2[:, col], np.ones(100)/100, mode='valid')
            ax.plot(data2[(n-1):, 0], ma2, label=label2)
            ax.legend()
    
    else: 
        ax.plot(data1[:, 0], data1[:, col], label=label1)
        if path2 is not None: 
            ax.plot(data2[:, 0], data2[:, col], label=label2)

    #ax.set_title("Average Loss per Episode")
    ax.set_ylabel(cols[col])
    ax.set_xlabel("Iterations")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    plt.savefig(saveas)


if __name__ == '__main__':
    plot_data(col=1, saveas='rewards_ma100', path1='results/CNN_1_data.txt', label1='Convolutional Model', path2='results/lin_1_data.txt', label2='Linear Model', n=100)
