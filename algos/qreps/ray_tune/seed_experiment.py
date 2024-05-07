import pandas as pd
from train_func_elbe import tune_elbe
from train_func_saddle import tune_saddle
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/nicolasvila/workplace/uni/tfg_v2/tests/tune_results_hebo_512.csv')
df.info()
for index, row in df.iterrows():
   if row['reward'] > 150:
    config = {}
    for column in df.columns:
        if column != 'config/__trial_index__':
          if column.startswith('config/'):
              field_name = column.split('/', 1)[1] 
              field_value = row[column] 
              config[field_name] = field_value

    print(config)
    rewards = []
    num_seeds = 3

    for seed in range(1, num_seeds + 1):
        config['seed'] = seed
        config['save_learning_curve'] = True
        config['eta'] = None
        try:
          rewards_df = tune_elbe(config)
          rewards.append(rewards_df)
        except:
           pass


    result = pd.concat(rewards)
    average_reward = result.groupby('Step')['Reward'].mean()
    rolling_average = average_reward.rolling(window=10).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(average_reward.index, average_reward, label='Original Reward', color='gray', alpha=0.7)
    plt.plot(rolling_average.index, rolling_average, label='Rolling Average (Window=7)', color='blue')

    plt.title('Average Episodic Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.savefig(f'./ray_tune/plots/plot_{row["trial_id"]}_{num_seeds}.png')
    plt.close()
