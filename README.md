# Warehouse Drones
MSc Project - A Comparative Analysis of Reward Functions for Reinforcement learning in a Warehouse Environment

This project presents an exploration of reinforcement learning (RL) agents' behaviours and performance in a warehouse automation environment. The main objective was to analyse the impact of different reward functions on the agents' learning processes and collaboration in a multi-agent system. The project utilized OpenAI's Gymnasium and PettingZoo libraries to develop the warehouse environment, enabling agents to navigate, pick up packages, and deliver them to specified goal states.

Two distinct reward functions were examined: sparse and dense rewards. The sparse reward function provided minimal feedback to agents, rewarding them only upon successful package deliveries. In contrast, the dense reward function offered more frequent and incremental feedback, guiding agents through a series of steps towards their goals. To assess the effects of these reward functions, the agents were trained independently, and their performance was compared over multiple epochs.

The results revealed that the sparse reward function allowed for faster convergence, with agents achieving an average score of 4 points out of 8. Conversely, the dense reward function led to higher overall performance, with agents reaching an average score of 36 out of 40. However, the dense reward function also introduced challenges related to volatility and potential suboptimal outcomes.

This project provides valuable insights into the impact of reward functions on RL agents' learning behaviours and collaboration in warehouse automation environments. The findings serve as a foundation for future research to enhance the adaptability and performance of RL agents in complex and dynamic scenarios.

Video Explanation at: [https://drive.google.com/file/d/1M1khW3AflVkJv29091FBFuwIBCy50eZv/view?usp=drive_link]
