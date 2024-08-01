import glob
import os
from stable_baselines3 import PPO
from pettingzoo.utils import parallel_to_aec

def evaluate(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    env = env_fn(render_mode=render_mode, **env_kwargs)
    env = parallel_to_aec(env)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    try:

        pattern = f"{env.metadata['num_bidders']}_provider_model*.zip"
        policy_files = glob.glob(os.path.join('models', pattern))

        print(policy_files)

        latest_policy = max(policy_files, key=os.path.getctime)

    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _ = env.last()
            rewards[agent] += reward

            if not termination or not truncation:
                if agent == env.agents[-1]: env.render()

            if termination or truncation:
                env.render()
                break

            else:
                act = model.predict(obs, deterministic=True)[0]
                #print(act, agent)

            env.step(act)

        if not termination or not truncation:env.render()

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward