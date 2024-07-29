from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy

def create_ppo_model(env, learning_rate, batch_size, ent_coef):
    return PPO(
        MultiInputPolicy,
        env,
        verbose=3,
        learning_rate=learning_rate,
        batch_size=batch_size,
        ent_coef=ent_coef,
    )