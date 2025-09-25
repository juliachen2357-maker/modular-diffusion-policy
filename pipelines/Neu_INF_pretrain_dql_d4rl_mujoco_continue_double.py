import os
import re
import csv
import gym
import d4rl
import hydra
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.utils import DQLCritic, report_parameters
from utils import set_seed
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dict = defaultdict(dict)

def extract_step(filename):
    match = re.search(r"_(\d+)\.pt$", filename)
    if match:
        return int(match.group(1))
    return None

def get_ckpts_with_fallback(guidance_dir, fallback_dir):
    diffusion_ckpts = {
        extract_step(f): os.path.join(guidance_dir[0], f)
        for f in os.listdir(guidance_dir[0])
        if f.startswith("diffusion_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }

    critic_ckpts = {
        extract_step(f): os.path.join(guidance_dir[1], f)
        for f in os.listdir(guidance_dir[1])
        if f.startswith("critic_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }

    fallback_candidates = [f for f in os.listdir(fallback_dir) if f.endswith(".pt")]
    assert len(fallback_candidates) == 1, f"❌ Expected 1 fallback .pt in {fallback_dir}, found: {fallback_candidates}"
    fallback_critic = os.path.join(fallback_dir, fallback_candidates[0])

    step_pairs = []
    for step, diff_path in diffusion_ckpts.items():
        critic_path = critic_ckpts.get(step, fallback_critic)
        step_pairs.append((step, diff_path, critic_path))
    return sorted(step_pairs)

def record_result(guidance_name, step, score, std, pipeline_name, env_name):
    results_dict[step][guidance_name] = (score, std)
    log_csv = f"log_results_{pipeline_name}_{env_name}_{timestamp}_continue_double.csv"
    file_exists = os.path.exists(log_csv)
    with open(log_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "step", "guidance", "mean", "variance"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            step,
            guidance_name,
            score,
            std ** 2
        ])
    print(f"✅ [Logged] step={step}, {guidance_name}, score={score:.4f}, var={std**2:.2e}")

def save_results_to_csv(results_dict, pipeline_name, env_name):
    steps = sorted(results_dict.keys())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = f"guidance_scores_{pipeline_name}_{env_name}_{timestamp}_continue_double.csv"
    all_guidances = sorted({g for step in results_dict for g in results_dict[step].keys()})
    data = {"step": steps}
    for g in all_guidances:
        data[f"{g}_mean"] = [results_dict[step].get(g, (None,))[0] for step in steps]
        data[f"{g}_variance"] = [results_dict[step].get(g, (None, None))[1]**2 if results_dict[step].get(g) else None for step in steps]
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"✅ Results saved to: {output_csv}")
def count_files(directory):
    return sum(1 for entry in os.scandir(directory) if entry.is_file())
def get_guidance_dirs(base_path):
    p1=[os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and "42" in d  ]
    p2=[os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and "42" not in d ]
    r1 = next((ele for ele in p1 if count_files(ele) > 1000), None)
    r2 = next((ele for ele in p2 if count_files(ele) > 1000), None)   
    return [r1,r2 ]
def load_ckpt_and_inference(guidance_name, step, actor, critic, critic_target,
                             diffusion_ckpt, critic_ckpt, args, dataset, env_eval):
    actor.load(diffusion_ckpt)
    env = gym.make(env_eval.env_name)
    critic_state = torch.load(critic_ckpt, map_location=args.device)
    critic.load_state_dict(critic_state["critic"])
    critic_target.load_state_dict(critic_state["critic_target"])
    actor.eval(), critic.eval(), critic_target.eval()

    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim
    normalizer = dataset.get_normalizer()
    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
    episode_rewards = []

    obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
    while not np.all(cum_done) and t < 1001:
        obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
        obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

        act, _ = actor.sample(prior, solver=args.solver, n_samples=args.num_envs * args.num_candidates,
                              sample_steps=args.sampling_steps, condition_cfg=obs, w_cfg=1.0,
                              use_ema=args.use_ema, temperature=args.temperature)

        with torch.no_grad():
            q = critic_target.q_min(obs, act)
            q = q.view(-1, args.num_candidates, 1)
            w = torch.softmax(q * args.task.weight_temperature, 1)
            act = act.view(-1, args.num_candidates, act.shape[-1])
            indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
            sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

        obs, rew, done, _ = env_eval.step(sampled_act)
        t += 1
        cum_done = np.logical_or(cum_done, done)
        ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
        if t % 100 == 0:
            print(diffusion_ckpt)
            print(f"{guidance_name}, t={t}, ep_reward={ep_reward}")
    episode_rewards.append(ep_reward)
    episode_rewards = np.array([env.get_normalized_score(r) for r in episode_rewards])
    mean_score = episode_rewards.mean()
    mean_std = episode_rewards.std()
    print(f"🎯 step {step} score: {mean_score:.4f}")
    record_result(guidance_name, step, mean_score, mean_std, args.pipeline_name, args.task.env_name)

@hydra.main(config_path="../configs/dql/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path1 = f"results/{args.pipeline_name}/{args.task.env_name}_pretrain/"
    fallback_dir = f"results/{args.pipeline_name}/pretrained"

    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    actor = DiscreteDiffusionSDE(nn_diffusion, nn_condition, predict_noise=args.predict_noise,
                                 optim_params={"lr": args.actor_learning_rate},
                                 x_max=torch.ones((1, act_dim), device=args.device),
                                 x_min=-torch.ones((1, act_dim), device=args.device),
                                 diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate,
                                 device=args.device)

    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()

    env_eval = gym.vector.make(args.task.env_name, args.num_envs)
    env_eval.env_name = args.task.env_name

    guidance_dirs = sorted(get_guidance_dirs(save_path1)[0:2])
    # for g_dir in guidance_dirs:
    guidance_name = "contiune_double"
    print(f"🔍 Evaluating: {guidance_name}")
    step_ckpt_pairs = sorted(get_ckpts_with_fallback(guidance_dirs, fallback_dir))
    if not step_ckpt_pairs:
        print(f"⚠️ No ckpt found in {guidance_name}")
        # continuestep_ckpt_pair
    for step, diff_ckpt, critic_ckpt in step_ckpt_pairs:
        if step<100000:
            load_ckpt_and_inference(guidance_name, step, actor, critic, critic_target,
                                diff_ckpt, critic_ckpt, args, dataset, env_eval)
    save_results_to_csv(results_dict, args.pipeline_name, args.task.env_name)

if __name__ == "__main__":
    pipeline()
