import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import glob
import os
import re
import csv
import gym
import d4rl
import hydra
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import IDQLQNet, IDQLVNet
from utils import set_seed
import pandas as pd
import glob, os, re,sys
results_dict = defaultdict(dict)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_path = os.path.abspath(sys.argv[0])

print("===== Script content starts =====")
with open(script_path, 'r') as f:
    print(f.read())
print("===== Script content ends =====")
def extract_step(filename):
    match = re.search(r"_(\d+)\.pt$", filename)
    return int(match.group(1)) if match else None

def get_ckpts_with_fallback(guidance_dir, fallback_dir):
    diffusion_ckpts = {
        extract_step(f): os.path.join(guidance_dir, f)
        for f in os.listdir(guidance_dir)
        if f.startswith("diffusion_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }

    critic_ckpts = {
        extract_step(f): os.path.join(guidance_dir, f)
        for f in os.listdir(guidance_dir)
        if f.startswith("iql") and f.endswith(".pt") and extract_step(f) is not None
    }

    fallback_candidates = [f for f in os.listdir(fallback_dir) if f.endswith(".pt")]
    assert len(fallback_candidates) == 1, f"Expected 1 fallback .pt in {fallback_dir}"
    fallback_critic = os.path.join(fallback_dir, fallback_candidates[0])

    step_pairs = []
    for step, diff_path in diffusion_ckpts.items():
        critic_path = critic_ckpts.get(step, fallback_critic)
        step_pairs.append((step, diff_path, critic_path))
    return sorted(step_pairs,reverse=True)

def record_result(guidance_name, step, score, std, args):
    results_dict[step][guidance_name] = (score, std)
    
    # output_csv = f"final_results_{pipeline_name}_{env_name}_{timestamp}.csv"
    log_csv = f"log_results_{args.pipeline_name}_{args.task.env_name}_{timestamp}.csv"
    file_exists = os.path.exists(log_csv)
    
    with open(log_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "step", "guidance", "mean", "std"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            step,
            guidance_name,
            score,
            std
        ])
    print(f"✅ Results logged - step:{step}, {guidance_name}, score:{score:.4f}±{std:.4f}")

def save_results_to_csv(results_dict, pipeline_name, env_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = f"final_results_{pipeline_name}_{env_name}_DL_idiffuser_dclassifier.csv"
    
    data = {"step": sorted(results_dict.keys())}
    for step in results_dict:
        for guidance in results_dict[step]:
            if f"{guidance}_mean" not in data:
                data[f"{guidance}_mean"] = []
                data[f"{guidance}_std"] = []
    
    for step in sorted(results_dict.keys()):
        for guidance in data:
            if guidance.endswith("_mean"):
                base_name = guidance[:-5]
                if base_name in results_dict[step]:
                    data[guidance].append(results_dict[step][base_name][0])
                    data[f"{base_name}_std"].append(results_dict[step][base_name][1])
                else:
                    data[guidance].append(None)
                    data[f"{base_name}_std"].append(None)
    
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"✅ Final results saved to {output_csv}")
def get_latest(iql_files):
    valid_files = []
    for f in iql_files:
        try:
            num = int(f.split('_')[-1].split('.')[0])
            valid_files.append((num, f))
        except (IndexError, ValueError):
            continue
    print(max(valid_files, key=lambda x: x[0])[1])
    return max(valid_files, key=lambda x: x[0])[1]
def get_guidance_dirs(base_path):
    return [os.path.join(base_path, d) for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and "guidance" in d]
def find_dql_ckpt(base_path,path):
    guidance_dirs = (glob.glob(os.path.join(base_path, path, "0_guidance_*")) +
                    glob.glob(os.path.join(base_path, path, "guidance_0_*")))
    latest_guidance = guidance_dirs[0]
    files = glob.glob(os.path.join(guidance_dirs[0], "critic_ckpt_*.pt"))
    return get_latest(files)
def find_diffusion_ckpt(base_path,path):
    guidance_dirs = (glob.glob(os.path.join(base_path, path, "0_guidance_*")) +
                    glob.glob(os.path.join(base_path, path, "guidance_0_*")))
    diffusion_files = glob.glob(os.path.join(guidance_dirs[0], "diffusion_ckpt_*.pt"))
    return get_latest(diffusion_files) 
def load_ckpt_and_inference( actor, critic, critic_target,args, dataset, env_eval):
    base_path = "results"
    Q_path = f"dql_d4rl_mujoco/{args.task.env_name}_pretrain"
    D_path = f"idql_d4rl_mujoco/{args.task.env_name}_pretrain"
    
    
    env = gym.make(env_eval.env_name)
    Q_ckpt = find_dql_ckpt(base_path,Q_path)
    D_ckpt = find_diffusion_ckpt(base_path,D_path)
    print(Q_ckpt,D_ckpt)



    actor.load(D_ckpt)
    critic_ckpt = torch.load(Q_ckpt, map_location=args.device)
    critic.load_state_dict(critic_ckpt["critic"])
    critic_target.load_state_dict(critic_ckpt["critic_target"])
    actor.eval()
    critic.eval()
    critic_target.eval()

    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim
    normalizer = dataset.get_normalizer()
    prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
    episode_rewards = []
    # for _ in range(args.num_episodes):
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
            act = act.view(-1, args.num_candidates, act_dim)

            indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
            sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
        t += 1
        obs, rew, done, info = env_eval.step(sampled_act)
        ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
        if t%100==0:
            print(f"t={t},ep_reward={ep_reward}")
    episode_rewards.append(ep_reward)

    episode_rewards = np.array([env.get_normalized_score(r) for r in episode_rewards])
    mean_score = episode_rewards.mean()
    mean_var = episode_rewards.var()
    print(f"🎯  score: {mean_score:.4f}")
    #record_result(guidance_name, step, mean_score,mean_var)
    # record_result(guidance_name, step, mean_score, mean_var, args)
def get_matching_ckpts(diffusion_dir, critic_dir):
    diffusion_ckpt_pattern = re.compile(r'(^|_)42_guidance_0(_|$)')


    diffusion_ckpt_filtered_items = [
        name for name in os.listdir(diffusion_dir)
        if diffusion_ckpt_pattern.search(name)
    ]
    diffusion_ckpt_full_path = os.path.join(diffusion_dir, diffusion_ckpt_filtered_items[0])

    critic_ckpts_pattern = re.compile(r'(^|_)42_guidance_0(_|$)')


    critic_ckpts_filtered_items = [
        name for name in os.listdir(critic_dir)
        if critic_ckpts_pattern.search(name)
    ]
    critic_ckpts_full_path = os.path.join(critic_dir, critic_ckpts_filtered_items[0])
    diffusion_ckpts = {
        extract_step(f): os.path.join(diffusion_ckpt_full_path, f)
        for f in os.listdir(diffusion_ckpt_full_path)
        if f.startswith("diffusion_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }
    critic_ckpts = {
        extract_step(f): os.path.join(critic_ckpts_full_path, f)
        for f in os.listdir(critic_ckpts_full_path)
        if f.startswith("critic_ckpt_") and f.endswith(".pt") and extract_step(f) is not None
    }
    common_steps = sorted(set(diffusion_ckpts.keys()) & set(critic_ckpts.keys()))
    matched_pairs = [(step, diffusion_ckpts[step], critic_ckpts[step]) for step in common_steps]
    return matched_pairs

@hydra.main(config_path="../configs/diql/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    C_save_path = f"results/dql_d4rl_mujoco/{args.task.env_name}_pretrain/"
    D_save_path = f"results/idql_d4rl_mujoco/{args.task.env_name}_pretrain/"

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

    
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    actor = DiscreteDiffusionSDE(
        nn_diffusion,
        nn_condition,
        predict_noise=args.predict_noise,
        optim_params={"lr": args.actor_learning_rate},
        x_max=torch.ones((1, act_dim), device=args.device),
        x_min=-torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps,
        ema_rate=args.ema_rate,
        device=args.device
    )

    iql_q = IDQLQNet(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
    iql_v = IDQLVNet(obs_dim, hidden_dim=args.critic_hidden_dim).to(args.device)

    env_eval = gym.vector.make(args.task.env_name, args.num_envs)
    env_eval.env_name = args.task.env_name



    guidance_dirs = get_matching_ckpts(D_save_path, C_save_path)
    guidance_name='TICDD'
    for step, diff_ckpt, critic_ckpt in guidance_dirs:
        if step < 100000:
            load_ckpt_and_inference((diff_ckpt, critic_ckpt), step, actor, iql_q, iql_q_target, iql_v,
                                 args, dataset, env_eval)


    # load_ckpt_and_inference( actor, iql_q, iql_q_target, iql_v,args, dataset, env_eval)
# load_ckpt_and_inference(guidance_name, step, actor, iql_q, iql_q_target, iql_v,
#                              diffusion_ckpt, iql_ckpt, args, dataset, env_eval)
    save_results_to_csv(results_dict, args.pipeline_name, args.task.env_name)








if __name__ == "__main__":
    pipeline()
