import argparse
import os
from distutils.util import strtobool
import time 
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
import numpy as np
import gymnasium as gym
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name",type=str,default=os.path.basename(__file__).rstrip(".py")
                        ,help="experiment name")
    parser.add_argument("--gym-id",type=str,default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate",type=float,default=2.5e-4,
                        help="learning rate")
    parser.add_argument("--seed",type=int,default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps",type=int,default=25000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help="if toggled,'torch.backends.cudnn.deterministic=False'")
    parser.add_argument("--cuda",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help='if toggled,cuda will not be enabled by default')
    parser.add_argument("--track",type=lambda x:bool(strtobool(x)),default=False,nargs='?',const=True,
                        help="if toggled,the experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project",type=str,default="ppo",
                        help="the name of the wandb project")
    parser.add_argument("--wandb-entity",type=str,default=None,
                        help="the name of the wandb's project entity")
    parser.add_argument("--capture-video",type=lambda x:bool(strtobool(x)),default=False,nargs='?',const=True,
                        help="weather to capture video of the agent performance (check out 'videos' folder)")
    parser.add_argument("--num-envs",type=int,default=4,
                        help="number of parallel environments")
    parser.add_argument("--num-steps",type=int,default=128,
                        help="number of steps to run in each environment per update")
    parser.add_argument("--anneal-lr",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help="if toggled, the learning rate will be annealed to 0")
    parser.add_argument("--gae",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help="Use GAE for advantage estimation")
    parser.add_argument("--gamma",type=float,default=0.99,
                        help="discount factor")
    parser.add_argument("--gae-lambda",type=float,default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches",type=int,default=4,
                        help="number of minibatches to split the batch into")
    parser.add_argument("--update-epochs",type=int,default=4,
                        help="number of epochs to update the policy")
    parser.add_argument("--norm-adv",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help="if toggled, the advantages will be normalized")
    parser.add_argument("--clip-coef",type=float,default=0.2,
                        help="the clip coefficient for PPO")
    parser.add_argument("--clip-vloss",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help="if toggled, the value loss will be clipped")
    parser.add_argument("--ent-coef",type=float,default=0.01,
                        help="the coefficient for the entropy loss")
    parser.add_argument("--vf-coef",type=float,default=0.5,
                        help = "the coefficient for the value loss")
    parser.add_argument("--max-grad-norm",type=float,default=0.5,
                        help="the maximum norm for the gradients clipping")
    parser.add_argument("--target-kl",type = float,default=None,
                        help="the target KL divergence")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args
def make_env(gym_id,seed,idx,capture_video,run_name):
        def thunk():
            env = gym.make(gym_id,render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx==0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 100 == 0)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
def layer_init(layer,std=np.sqrt(2),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer
class Agent(nn.Module):
    def __init__(self, envs):
          super(Agent,self).__init__()
          self.critic = nn.Sequential(
               layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,1),std=1.0)
          )
          self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,envs.single_action_space.n),std=0.01)
          )
    def get_value(self,x):
        return self.critic(x)
    def get_action_and_value(self,x,action=None):
         logits = self.actor(x)
         probs = Categorical(logits=logits)
         if action is None:
             action = probs.sample()
         return action,probs.log_prob(action),probs.entropy(),self.critic(x)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
   
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id,args.seed+i,i,args.capture_video,run_name) 
                                     for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(),lr=args.learning_rate,eps=1e-5)
    obs = torch.zeros((args.num_steps,args.num_envs)+envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,args.num_envs)+envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps,args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps,args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps,args.num_envs)).to(device)
    values = torch.zeros((args.num_steps,args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs =torch.tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    for update in range(1,num_updates+1):
        if args.anneal_lr:
              frac = 1.0 - (update - 1) / num_updates
              lrnow = args.learning_rate * frac
              optimizer.param_groups[0]["lr"]=lrnow
        for step in range(0,args.num_steps):
             global_step += 1*args.num_envs
             obs[step] = next_obs
             dones[step] = next_done
             with torch.no_grad():
                  action,logprob,_,value = agent.get_action_and_value(next_obs)
                  values[step] = value.flatten()
             actions[step] = action
             logprobs[step] = logprob

             next_obs,reward,terminated,truncated,info = envs.step(action.cpu().numpy())
             next_done = terminated | truncated
             rewards[step] = torch.tensor(reward).to(device).view(-1)
             next_obs,next_done = torch.tensor(next_obs).to(device),torch.tensor(next_done).to(device)

             if "episode" in info.keys():
                     print(f"global_step={global_step}, episodic_return={sum(info['episode']['r'])}")
                     writer.add_scalar("charts/episodic_return", sum(info["episode"]["r"].item()), global_step)
                     writer.add_scalar("charts/episodic_length", sum(info["episode"]["l"].item()), global_step)
             
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1,-1)
            if args.gae:
                 advantages = torch.zeros_like(rewards).to(device)
                 lastgaelam = 0
                 for i in reversed(range(args.num_steps)):
                    if i == args.num_steps-1:
                          nextnonterminal = ~next_done
                          nextvalues = next_value 
                    else:
                           nextnonterminal = 1.0 - dones[i+1]
                           nextvalues = values[i+1]
                    delta = rewards[i]+args.gamma*nextvalues*nextnonterminal - values[i]
                    advantages[i] = lastgaelam = delta + args.gamma*args.gae_lambda*nextnonterminal*lastgaelam
                 returns = advantages + values
            else:
                 advantages = torch.zeros_like(rewards).to(device)
                 returns = torch.zeros_like(rewards).to(device) 
                 for i in reversed(range(args.num_steps)):
                     if i == args.num_steps-1:
                         nextnonterminal = ~next_done
                         nextvalues = next_value
                     else:
                         nextnonterminal = 1.0 - dones[i+1]
                         nextvalues = values[i+1]
                     returns[i] = rewards[i] + args.gamma * args.gae_lambda*nextvalues * nextnonterminal
                 advantages = returns - values
        b_obs = obs.reshape((-1,)+envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,)+envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []    
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0,args.batch_size,args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _,newlogprob,entropy,new_values = agent.get_action_and_value(
                    b_obs[mb_inds],b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio-1)-logratio).mean()
                    clipfracs += [((ratio-1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio,1-args.clip_coef,1+args.clip_coef)
                pg_loss = torch.max(pg_loss1,pg_loss2).mean()

                newvalues = new_values.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalues - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                      new_values-b_values[mb_inds],
                      -args.clip_coef,
                      args.clip_coef
                 )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_unclipped,v_loss_clipped)
                    v_loss = v_loss_max.mean()*0.5
                else:
                    v_loss = 0.5*((newvalues-b_returns[mb_inds])**2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        y_pred,y_true = b_values.cpu().numpy(),b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true-y_pred)/var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    envs.close()
    writer.close()
    
                     
                    
                 
                  
             
             