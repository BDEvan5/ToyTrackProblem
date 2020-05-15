from matplotlib import pyplot as plt
import numpy as np

# plotting utils 
def plot(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 5, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    # print("Episode", (len(values)), "\n", \
    #     moving_avg_period, "episode moving avg:", moving_avg)

def plot_comp(values1, values2, moving_avg_period, title, figure_n):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    # plt.plot(values1)

    moving_avg = get_moving_average(moving_avg_period, values1)
    plt.plot(moving_avg)    

    # plt.plot(values2)
    moving_avg = get_moving_average(moving_avg_period, values2)
    plt.plot(moving_avg)    
 
    plt.legend(['RL Moving Avg', "Classical Moving Avg"])
    # plt.legend(['RL Agent', 'RL Moving Avg', 'Classical Agent', "Classical Moving Avg"])
    plt.pause(0.001)

def get_moving_average(period, values):

    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg


