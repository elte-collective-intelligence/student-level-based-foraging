import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/home/kwamboka/dqn_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/resource/goalcollisions.log' 
df = pd.read_csv(file_path)

episode_10_data = df[df['Episode'] == 10]

num_agents = episode_10_data['Num_Agents'].unique()
stage = episode_10_data['Stage'].values[0]

goals = []
collisions = []
for agents in num_agents:
    agent_data = episode_10_data[episode_10_data['Num_Agents'] == agents]
    goals.append(agent_data['Goals'].values[0])
    collisions.append(agent_data['Collisions'].values[0])

x = np.arange(len(num_agents)) 
width = 0.35  

fig, ax = plt.subplots(figsize=(10, 6))
bars_goals = ax.bar(x - width / 2, goals, width, label='Goals', color='green')
bars_collisions = ax.bar(x + width / 2, collisions, width, label='Collisions', color='red')

ax.set_xlabel('Number of Agents')
ax.set_ylabel('Counts')
ax.set_title(f'Goals vs Collisions (Episode 10, Stage {stage})')
ax.set_xticks(x)
ax.set_xticklabels(num_agents)
ax.legend()

for bars in [bars_goals, bars_collisions]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
