import gymnasium as gym
import matplotlib.pyplot as plt

# Set up the environment
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

plt.ion()  # Turn on interactive mode for non-blocking plotting
fig, ax = plt.subplots()
xdata, ydata = [], []

# Initialize the plot
ax.set_xlim(0, 1000)
ax.set_ylim(-500, 300)  # Adjust the y-axis limits according to expected rewards
line, = ax.plot(xdata, ydata, 'r-', label='Reward per Episode')
ax.set_title('Reward Graph')  # Title for the graph
ax.set_xlabel('Episode')  # X-axis label
ax.set_ylabel('Total Reward')  # Y-axis label
ax.legend()  # Add a legend

def update_plot(x, y):
    line.set_xdata(x)
    line.set_ydata(y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

# Run the game loop
total_reward = 0
for i in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        xdata.append(i)
        ydata.append(total_reward)
        update_plot(xdata, ydata)  # Update the plot with the latest reward

        total_reward = 0
        observation, info = env.reset()  # Reset the environment

env.close()  # Close the environment when done