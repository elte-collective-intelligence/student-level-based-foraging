import pandas as pd
import matplotlib.pyplot as plt

def plot_avg_reward_vs_agents(log_file):

    data = pd.read_csv(log_file)

    required_columns = {'Stage', 'Episode', 'Agent', 'Cumulative_Reward'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Log file must include columns: {required_columns}")


    avg_rewards_per_agent = (
        data.groupby(['Stage', 'Agent'])['Cumulative_Reward']
        .mean()
        .reset_index()
    )


    avg_rewards_by_stage = (
        avg_rewards_per_agent.groupby('Stage')['Cumulative_Reward']
        .mean()
        .reset_index()
    )

    stages = avg_rewards_by_stage['Stage'].to_numpy()
    avg_rewards = avg_rewards_by_stage['Cumulative_Reward'].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(stages, avg_rewards, 'o-', label='Average Reward')

    plt.title("Average Reward per Agent vs. Number of Agents", fontsize=14)
    plt.xlabel("Number of Agents (Stage)", fontsize=12)
    plt.ylabel("Average Reward per Agent", fontsize=12)
    plt.grid(ls=":")
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.show()

log_file_path = "/home/kwamboka/dqn_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/rewards.log"  
plot_avg_reward_vs_agents(log_file_path)
