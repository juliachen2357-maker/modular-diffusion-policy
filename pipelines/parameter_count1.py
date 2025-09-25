import os
import re
import torch
import gym
import d4rl
import hydra
from copy import deepcopy
from datetime import datetime
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.utils import DQLCritic
from utils import set_seed

# ====================== Newly added: model parameter statistics ======================
def count_parameters(model, model_name=""):
    """Count the number of parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{model_name}:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Memory usage: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    print("-" * 50)
    
    return total_params, trainable_params

def analyze_model_parameters(nn_diffusion, nn_condition, critic, critic_target, mode="Analysis"):
    """Analyze model parameter scale"""
    print("\n" + "="*60)
    print(f"Model Parameter Analysis - {mode}")
    print("="*60)
    
    # Count parameters for each model
    diffusion_total, diffusion_trainable = count_parameters(nn_diffusion, "Diffusion network (nn_diffusion)")
    condition_total, condition_trainable = count_parameters(nn_condition, "Condition network (nn_condition)")
    critic_total, critic_trainable = count_parameters(critic, "Critic model")
    
    # Calculate totals
    actor_total = diffusion_total + condition_total
    total_params = actor_total + critic_total
    total_trainable = diffusion_trainable + condition_trainable + critic_trainable
    
    print("\nSummary:")
    print(f"Diffusion network: {diffusion_total:,} ({diffusion_total/total_params*100:.1f}%)")
    print(f"Condition network: {condition_total:,} ({condition_total/total_params*100:.1f}%)")
    print(f"Critic: {critic_total:,} ({critic_total/total_params*100:.1f}%)")
    print(f"Actor total: {actor_total:,} ({actor_total/total_params*100:.1f}%)")
    print(f"Model total: {total_params:,}")
    print(f"Trainable total: {total_trainable:,}")
    print(f"Estimated memory usage: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return {
        'diffusion_params': diffusion_total,
        'condition_params': condition_total,
        'actor_params': actor_total,
        'critic_params': critic_total, 
        'total_params': total_params,
        'diffusion_memory_mb': diffusion_total * 4 / 1024 / 1024,
        'condition_memory_mb': condition_total * 4 / 1024 / 1024,
        'actor_memory_mb': actor_total * 4 / 1024 / 1024,
        'critic_memory_mb': critic_total * 4 / 1024 / 1024,
        'total_memory_mb': total_params * 4 / 1024 / 1024
    }

def extract_step(filename):
    """Extract step number from filename"""
    match = re.search(r"_(\d+)\.pt$", filename)
    if match:
        return int(match.group(1))
    return None
# ====================== End of newly added code ======================

@hydra.main(config_path="../configs/dql/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    
    # Create models
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()

    # ====================== Newly added: analyze initial model parameters ======================
    print("\n" + "="*60)
    print("Initial model parameter analysis")
    print("="*60)
    model_info = analyze_model_parameters(nn_diffusion, nn_condition, critic, critic_target, "Initial model")
    
    # Save parameter info to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"model_analysis_{args.pipeline_name}_{args.task.env_name}_{timestamp}.txt"
    with open(save_path, "w") as f:
        f.write("Model Parameter Analysis Report\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Environment: {args.task.env_name}\n")
        f.write(f"Device: {args.device}\n\n")
        f.write(f"Diffusion network params: {model_info['diffusion_params']:,}\n")
        f.write(f"Condition network params: {model_info['condition_params']:,}\n")
        f.write(f"Actor total params: {model_info['actor_params']:,}\n")
        f.write(f"Critic params: {model_info['critic_params']:,}\n")
        f.write(f"Model total params: {model_info['total_params']:,}\n")
        f.write(f"Diffusion memory: {model_info['diffusion_memory_mb']:.2f} MB\n")
        f.write(f"Condition memory: {model_info['condition_memory_mb']:.2f} MB\n")
        f.write(f"Actor memory: {model_info['actor_memory_mb']:.2f} MB\n")
        f.write(f"Critic memory: {model_info['critic_memory_mb']:.2f} MB\n")
        f.write(f"Total memory: {model_info['total_memory_mb']:.2f} MB\n")
    
    print(f"\n✅ Parameter analysis saved to: {save_path}")
    # ====================== End of newly added code ======================

    # Optionally analyze saved models
    model_dir = f"results/{args.pipeline_name}/{args.task.env_name}/"
    if os.path.exists(model_dir):
        print(f"\n🔍 Checking model directory: {model_dir}")
        
        # Find latest model files
        diffusion_files = [f for f in os.listdir(model_dir) 
                          if f.startswith("diffusion_ckpt_") and f.endswith(".pt")]
        critic_files = [f for f in os.listdir(model_dir) 
                       if f.startswith("critic_ckpt_") and f.endswith(".pt")]
        
        if diffusion_files and critic_files:
            # Select latest checkpoint
            latest_diffusion = max(diffusion_files, key=extract_step)
            latest_critic = max(critic_files, key=extract_step)
            
            print(f"\n📁 Found saved models:")
            print(f"   Diffusion model: {latest_diffusion}")
            print(f"   Critic model: {latest_critic}")
            
            # Create actor instance to load models
            actor = DiscreteDiffusionSDE(
                nn_diffusion, nn_condition, predict_noise=args.predict_noise,
                optim_params={"lr": args.actor_learning_rate},
                x_max=torch.ones((1, act_dim), device=args.device),
                x_min=-torch.ones((1, act_dim), device=args.device),
                diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate,
                device=args.device)
            
            # Load checkpoints
            actor.load(os.path.join(model_dir, latest_diffusion))
            critic_state = torch.load(os.path.join(model_dir, latest_critic), map_location=args.device)
            critic.load_state_dict(critic_state["critic"])
            critic_target.load_state_dict(critic_state["critic_target"])
            
            # ====================== Newly added: analyze saved model parameters ======================
            print("\n" + "="*60)
            print("Saved model parameter analysis")
            print("="*60)
            # Get internal networks from actor
            nn_diffusion_loaded = actor.nn_diffusion
            nn_condition_loaded = actor.nn_condition
            
            saved_model_info = analyze_model_parameters(
                nn_diffusion_loaded, nn_condition_loaded, critic, critic_target, "Saved model")
            
            # Save report
            saved_save_path = f"saved_model_analysis_{args.pipeline_name}_{args.task.env_name}_{timestamp}.txt"
            with open(saved_save_path, "w") as f:
                f.write("Saved Model Parameter Analysis Report\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Environment: {args.task.env_name}\n")
                f.write(f"Device: {args.device}\n")
                f.write(f"Diffusion checkpoint: {latest_diffusion}\n")
                f.write(f"Critic checkpoint: {latest_critic}\n\n")
                f.write(f"Diffusion network params: {saved_model_info['diffusion_params']:,}\n")
                f.write(f"Condition network params: {saved_model_info['condition_params']:,}\n")
                f.write(f"Actor total params: {saved_model_info['actor_params']:,}\n")
                f.write(f"Critic params: {saved_model_info['critic_params']:,}\n")
                f.write(f"Model total params: {saved_model_info['total_params']:,}\n")
                f.write(f"Diffusion memory: {saved_model_info['diffusion_memory_mb']:.2f} MB\n")
                f.write(f"Condition memory: {saved_model_info['condition_memory_mb']:.2f} MB\n")
                f.write(f"Actor memory: {saved_model_info['actor_memory_mb']:.2f} MB\n")
                f.write(f"Critic memory: {saved_model_info['critic_memory_mb']:.2f} MB\n")
                f.write(f"Total memory: {saved_model_info['total_memory_mb']:.2f} MB\n")
            
            print(f"\n✅ Saved model parameter analysis saved to: {saved_save_path}")
            # ====================== End of newly added code ======================
        else:
            print("⚠️ No complete model checkpoints found")
    else:
        print("⚠️ Model directory not found")

if __name__ == "__main__":
    pipeline()
