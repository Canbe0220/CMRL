import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from multimodal import CDP_CNNs, HTB
from mlp import MLPCritic, MLPActor

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []
        
        self.curr_proc_batch = []
        self.raw_image = []
        self.raw_mas = []
        self.values = []
        self.eligible = []
        
        
    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        
        del self.curr_proc_batch[:]
        del self.raw_image[:]
        del self.raw_mas[:]
        del self.values[:]
        del self.eligible[:]

class CMRL(nn.Module):
    def __init__(self, model_paras):
        super(CMRL, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]
        self.num_mas = model_paras["num_mas"]

        self.get_operations = CDP_CNNs()
        self.get_machines = HTB((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads, self.dropout, self.dropout)
        

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)


    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_image = state.feat_image_batch[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        curr_proc_batch = state.curr_proc_batch[batch_idxes]

        num_mas = raw_mas.size(-2)
        h_jobs = self.get_operations(raw_image, num_mas)
        h_mas = self.get_machines(curr_proc_batch[..., 0], (h_jobs, raw_mas))
        h_pairs = curr_proc_batch

        # Stacking and pooling
        h_jobs_pooled = h_jobs.max(dim=-2)[0]
        h_mas_pooled = h_mas.max(dim=-2)[0]
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)

        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_jobs_pooled_padding = h_jobs_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = (curr_proc_batch[..., 0] == 1) & job_eligible & ma_eligible

        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        # interaction = h_jobs_padding * h_mas_padding
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_pairs),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_jobs_pooled, h_mas_pooled), dim=-1)
        values = self.critic(h_pooled)
        mask = eligible.transpose(1, 2).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        # Store data in memory during training
        if flag_train == True:
            memories.curr_proc_batch.append(copy.deepcopy(state.curr_proc_batch))
            memories.raw_image.append(copy.deepcopy(raw_image))
            memories.raw_mas.append(copy.deepcopy(raw_mas))
            memories.eligible.append(copy.deepcopy(eligible))
            memories.values.append(copy.deepcopy(values.squeeze()))

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, curr_proc_batch, raw_image, raw_mas, eligible, action_envs, flag_sample=False):
        
        num_mas = raw_mas.size(-2)
        h_jobs = self.get_operations(raw_image, num_mas)
        h_mas = self.get_machines(curr_proc_batch[..., 0], (h_jobs, raw_mas))
        h_pairs = curr_proc_batch

        # Stacking and pooling
        h_jobs_pooled = h_jobs.max(dim=-2)[0]
        h_mas_pooled = h_mas.max(dim=-2)[0]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        # h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, h_mas.size(-2), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_jobs_pooled_padding = h_jobs_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_pairs),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_jobs_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze(), dist_entropys


class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = CMRL(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory, env_paras, train_paras):
        
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]
        lam = train_paras.get("gae_lambda", 0.99) # GAE λ

        old_curr_proc_batch = torch.stack(memory.curr_proc_batch, dim=0).transpose(0, 1)
        old_raw_image = torch.stack(memory.raw_image, dim=0).transpose(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)      
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1)
        old_values_envs = torch.stack(memory.values, dim=0).transpose(0, 1)

        num_envs, steps = memory_rewards.shape

        # Computation of Monte Carlo returns
        rewards_envs_mc = []
        initial_discounted_rewards_log = 0.0
        for i in range(num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward)
            initial_discounted_rewards_log += discounted_reward
            rewards_envs_mc.append(torch.tensor(rewards, dtype=torch.float32, device=device))
        # rewards_envs_mc for logging purposes and excluded from model updates

        
        # GAE and Returns
        old_values_envs = old_values_envs.detach()
        advantages = []
        returns = []
        for i in range(num_envs):
            rew = memory_rewards[i]                         
            term = memory_is_terminals[i]
            val = old_values_envs[i]                         

            gae = 0.0
            adv_list = []
            
            gae = 0.0
            for t in reversed(range(steps)):
                done = term[t].float()
                if t == steps - 1:
                    next_value = 0.0
                else:
                    next_value = val[t + 1]

                delta = rew[t] + self.gamma * next_value * (1 - done) - val[t]
                gae = delta + self.gamma * lam * (1 - done) * gae
                adv_list.insert(0, gae)                      

            adv_tensor = torch.tensor(adv_list, dtype=torch.float32, device=device)
            ret_tensor = adv_tensor + val                     # returns = advantages + values
            advantages.append(adv_tensor)
            returns.append(ret_tensor)

        advantages = torch.stack(advantages)                 # [num_envs, steps]
        returns = torch.stack(returns)                       # [num_envs, steps]
        
        
        adv_flat = advantages.flatten()
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.view_as(advantages).detach()
        returns = returns.detach()                  

        # Flatten
        old_curr_proc_batch = old_curr_proc_batch.flatten(0, 1)
        old_raw_image = old_raw_image.flatten(0, 1)
        old_raw_mas = old_raw_mas.flatten(0, 1)
        old_eligible = old_eligible.flatten(0, 1)
        old_logprobs = old_logprobs.flatten(0, 1).detach()
        old_action_envs = old_action_envs.flatten(0, 1)
        advantages_flat = advantages.flatten(0, 1)             # [total_samples]
        returns_flat = returns.flatten(0, 1)                   # [total_samples]

        full_batch_size = old_curr_proc_batch.size(0)
        indices = torch.arange(full_batch_size)

        # Leveraging advantages and targets for multi-epoch PPO updates
        loss_epochs = 0.0
        for _ in range(self.K_epochs):
            shuffled_indices = indices[torch.randperm(full_batch_size)]
            for start_idx in range(0, full_batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size
                batch_indices = shuffled_indices[start_idx:end_idx]

                old_logp_batch = old_logprobs[batch_indices]
                adv_batch = advantages_flat[batch_indices]
                ret_batch = returns_flat[batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_curr_proc_batch[batch_indices],
                    old_raw_image[batch_indices],
                    old_raw_mas[batch_indices],
                    old_eligible[batch_indices],
                    old_action_envs[batch_indices]
                )

                ratios = torch.exp(logprobs - old_logp_batch)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_batch
                policy_loss = -self.A_coeff * torch.min(surr1, surr2)
                value_loss = self.vf_coeff * F.mse_loss(state_values, ret_batch)
            
                entropy_loss = -self.entropy_coeff * dist_entropy

                loss = policy_loss + value_loss + entropy_loss
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        num_updates = self.K_epochs * math.ceil(full_batch_size / minibatch_size)
        return loss_epochs.item() / num_updates, initial_discounted_rewards_log.item() / self.num_envs