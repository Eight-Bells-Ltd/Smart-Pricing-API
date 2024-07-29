  

# 🏆 Reverse Auction Environment

  

![Python Version](https://img.shields.io/badge/python-3.8.11-blue.svg)

  

This project implements a reverse auction environment using the PettingZoo library and trains agents using Stable Baselines 3. The environment simulates a reverse auction where multiple agents compete by submitting bids, with the goal of offering the lowest price while maintaining a profitable position.

  

## 🌟 Features

  

- Custom Reverse Auction Environment

- Training with Proximal Policy Optimization (PPO)

- Comprehensive agent evaluation

- Dynamic auction round visualization

- Automatic video generation of auction processes

  

## 🚀 Quick Start

  

### Prerequisites

  

- Anaconda or Miniconda

- Git

  

### Installation

  

1. Clone the repository:

```bash

git clone https://github.com/Eight-Bells-Ltd/Smart_Pricing_MARL_NANCY.git

```

2. Create and activate a virtual environment with Python 3.11.8:

```bash

conda create -n reverse_auction python=3.11.8

conda activate reverse_auction

cd Smart_Pricing_MARL_NANCY

```

3. Install the required packages:

```bash

pip install -r requirements.txt

```

  

## 🏃‍♂️ Usage

  

### Training

  

Initiate the training process:

  

```bash

python  main.py  --mode  train

```

This command launches the PPO algorithm to train your agents. The trained model will be saved in the `models` directory.

  

### Evaluation

  

Evaluate your trained agents:

  

```bash

python  main.py  --mode  evaluate

```

This loads the most recent model and runs evaluation episodes, printing results to the console.

### Configuration

You can adjust training parameters and environment settings via the `config.yml` file, such as learning rate, number of agents, and auction rounds.

  

## 🧪 Environment Details

  

The `ReverseAuctionEnv` is a custom implementation using PettingZoo's `ParallelEnv`. Key features include:

  

- Multiple bidding agents

- Discrete action space for bid adjustments

- Observations including current rank, previous rank, and current round

- Rewards based on rank improvements, bid values, and final positions


### Example Environment State

| Agent | Current Bid | Rank | Previous Rank | Round |
|-------|-------------|------|---------------|-------|
| A1    | $50         | 1    | 2             | 3     |
| A2    | $55         | 2    | 1             | 3     |
| A3    | $60         | 3    | 3             | 3     |


  

## 📊 Visualization

  

The environment includes a rendering function that creates plots of each auction round. These plots are saved as PNG files in the `outputs/pngs` directory.
  

After evaluation, a video is automatically generated from these PNG files, providing a visual representation of the auction process. This video is saved in the `outputs` directory.

  

## ⚙️ Customization

  

Fine-tune various parameters of the environment and training/evaluation process by editing the `config.yml` file.

  

## 🙏 Acknowledgements

  

This project uses the following open-source libraries:

- [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo)

- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

- [NumPy](https://numpy.org/)

- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://opencv.org/)
- [SuperSuit](https://github.com/PettingZoo-Team/SuperSuit)

## ✉️ Contact

For inquiries, please reach out to [ilias.theodoropoulos@8bellsresearch.com](ilias.theodoropoulos@8bellsresearch.com).
