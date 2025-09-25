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
import datetime
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.utils import DQLCritic, report_parameters
from utils import set_seed

results_dict = defaultdict(dict)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def extract_step(filename):
    match = re.search(r"_(\d+)\.pt$", filename)
    if match:
        return int(match.group(1))
    return None

def get_ckpts_with_fallback(guidance_dir, fallback_dir):
    diffusion_ckpts = {
        extract_step(f): os.path.join(guidance_dir, f)
        for f in os.listdir(guidance_dir)
        if f.startswith("diffusion_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }

    critic_ckpts = {
        extract_step(f): os.path.join(guidance_dir, f)
        for f in os.listdir(guidance_dir)
        if f.startswith("critic_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }

    # fallback_candidates = [f for f in os.listdir(fallback_dir) if f.endswith(".pt")]
    # assert len(fallback_candidates) == 1, f"❌ Expected 1 fallback .pt in {fallback_dir}, found: {fallback_candidates}"
    fallback_critic = ''#os.path.join(fallback_dir, fallback_candidates[0])

    step_pairs = []
    for step, diff_path in diffusion_ckpts.items():
        critic_path = critic_ckpts.get(step, fallback_critic)
        step_pairs.append((step, diff_path, critic_path))
    return sorted(step_pairs)

def record_result(guidance_name, step, score, std, pipeline_name, env_name, diffusion_ckpt, critic_ckpt):
    results_dict[step][guidance_name] = (diffusion_ckpt, critic_ckpt, score, std)
    
    log_csv = f"log_results_{pipeline_name}_{env_name}_{timestamp}_baseline.csv"
    file_exists = os.path.exists(log_csv)
    with open(log_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "step", "diffusion_ckpt", "critic_ckpt", "guidance", "mean", "variance"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            step,
            diffusion_ckpt,
            critic_ckpt,
            guidance_name,
            score,
            std ** 2
        ])
    print(f"✅ [Logged] step={step}, {guidance_name}, score={score:.4f}, var={std**2:.2e}")

def save_results_to_csv(results_dict, pipeline_name, env_name):
    steps = sorted(results_dict.keys())
    output_csv = f"guidance_scores_{pipeline_name}_{env_name}_{timestamp}_baseline.csv"
    all_guidances = sorted({g for step in results_dict for g in results_dict[step].keys()})
    
    data = {"step": steps}
    for g in all_guidances:
        data[f"{g}_mean"] = [results_dict[step].get(g, (None, None, None, None))[2] for step in steps]
        data[f"{g}_variance"] = [results_dict[step].get(g, (None, None, None, None))[3]**2 if results_dict[step].get(g) else None for step in steps]
    
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"✅ Results saved to: {output_csv}")

def get_guidance_dirs(base_path):
    return [base_path]

def load_ckpt_and_inference(guidance_name, step, actor, critic, critic_target,
                             diffusion_ckpt, critic_ckpt, args, dataset, env_eval):
    actor.load(diffusion_ckpt)
    
    if critic_ckpt:  # 检查 critic_ckpt 是否为空
        critic_state = torch.load(critic_ckpt, map_location=args.device)
        critic.load_state_dict(critic_state["critic"])
        critic_target.load_state_dict(critic_state["critic_target"])
    
    actor.eval()
    critic.eval()
    critic_target.eval()

    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim
    normalizer = dataset.get_normalizer()
    episode_rewards = []

    for i in range(args.num_episodes):
        prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 1000 + 1:
            # normalize obs
            obs_tensor = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            obs_tensor = obs_tensor.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

            # sample actions
            act, _ = actor.sample(
                prior,
                solver=args.solver,
                n_samples=args.num_envs * args.num_candidates,
                sample_steps=args.sampling_steps,
                condition_cfg=obs_tensor, w_cfg=1.0,
                use_ema=args.use_ema, temperature=args.temperature)

            # resample
            with torch.no_grad():
                q = critic_target.q_min(obs_tensor, act)
                q = q.view(-1, args.num_candidates, 1)
                w = torch.softmax(q * args.task.weight_temperature, 1)
                act = act.view(-1, args.num_candidates, act_dim)

                indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

            # step
            obs, rew, done, info = env_eval.step(sampled_act)
            
            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += rew
            
            if np.all(cum_done):
                break

        episode_rewards.append(ep_reward)

    env = gym.make(args.task.env_name)  # 创建单独的环境用于计算标准化分数
    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)
    mean_score = np.mean(episode_rewards)
    mean_std = np.std(episode_rewards)
    
    print(f"🎯 step {step} score: {mean_score:.4f} ± {mean_std:.4f}")
    record_result(guidance_name, step, mean_score, mean_std, args.pipeline_name, args.task.env_name, diffusion_ckpt, critic_ckpt)

@hydra.main(config_path="../configs/edp/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):
    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}_{args.seed}/{args.task.env_name}/'
    fallback_dir = f'results/{args.pipeline_name}_{args.seed}/{args.task.env_name}/pretrained'
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(fallback_dir, exist_ok=True)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=False, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

    # ------------------ Critic ---------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()

    # ---------------------- Inference ----------------------
    env_eval = gym.vector.make(args.task.env_name, args.num_envs)

    guidance_dirs = get_guidance_dirs(save_path)
    for g_dir in guidance_dirs:
        guidance_name = os.path.basename(g_dir)
        print(f"🔍 Evaluating: {guidance_name}")
        step_ckpt_pairs = get_ckpts_with_fallback(g_dir, fallback_dir)
        if not step_ckpt_pairs:
            print(f"⚠️ No ckpt found in {guidance_name}")
            continue
        
        # Sort by step in descending order (latest first)
        step_ckpt_pairs = sorted(step_ckpt_pairs, key=lambda x: x[0], reverse=True)
        
        for step, diff_ckpt, critic_ckpt in step_ckpt_pairs:
            load_ckpt_and_inference(guidance_name, step, actor, critic, critic_target,
                                    diff_ckpt, critic_ckpt, args, dataset, env_eval)
        save_results_to_csv(results_dict, args.pipeline_name, args.task.env_name)

if __name__ == "__main__":
    pipeline()