import time
import supersuit as ss
from stable_baselines3.common.callbacks import EvalCallback
from agents.ppo_agent import create_ppo_model
import os

def train(env_fn, steps: int = 10_000, learning_rate=1e-3, batch_size=256, ent_coef=0.5, seed: int | None = 0, **env_kwargs):
    env = env_fn(**env_kwargs)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 16, num_cpus=16, base_class="stable_baselines3")


    model = create_ppo_model(env, learning_rate, batch_size, ent_coef)

    model.learn(total_timesteps=steps)

    model_filename = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_path = os.path.join('models', model_filename)

    model.save(model_path)

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()