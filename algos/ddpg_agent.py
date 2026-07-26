import os
import sys

sys.path.append("../")
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer
from algos.her import her_sampler
from planner.goal_plan import *
import utils
import gymnasium as gym
import numpy as np
import torch


class ddpg_agent:
    def __init__(self, args, env, env_params, test_env):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")

        self.writer = None
        self.writer = SummaryWriter(
            log_dir="runs/ddpg" + current_time + "_" + str(args.env_name)
        )
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
            # path to save the model
        self.model_path = os.path.join(
            self.args.save_dir, self.args.env_name + "_" + current_time
        )
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.actor_network = actor(env_params)
        self.actor_target_network = actor(env_params)
        self.critic_network = criticWrapper(self.env_params, self.args)
        self.critic_target_network = criticWrapper(self.env_params, self.args)

        if self.args.plan_budget < 0:
            self.args.plan_budget = self.args.landmark

        self.start_epoch = 0
        if self.resume == True:
            self.start_epoch = self.resume_epoch
            self.actor_network.load_state_dict(
                torch.load(
                    self.args.resume_path
                    + "/actor_model_"
                    + str(self.resume_epoch)
                    + ".pt"
                )[0]
            )
            self.critic_network.load_state_dict(
                torch.load(
                    self.args.resume_path
                    + "/critic_model_"
                    + str(self.resume_epoch)
                    + ".pt"
                )[0]
            )

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # if use gpu
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        self.actor_target_network.to(self.device)
        self.critic_target_network.to(self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=self.args.lr_actor
        )
        self.critic_optim = torch.optim.Adam(
            self.critic_network.parameters(), lr=self.args.lr_critic
        )
        # her sampler
        self.her_module = her_sampler(
            self.args.replay_strategy,
            self.args.replay_k,
            self.args.distance,
            self.args.future_step,
        )
        # create the replay buffer
        fetch_task = True if args.env_name == "Pusher-v0" else False
        self.buffer = replay_buffer(
            self.env_params,
            self.args.buffer_size,
            self.her_module.sample_her_transitions,
            self.args.plan_budget,
            fetch_task=fetch_task,
        )
        self.planner_policy = Planner(
            agent=self,
            replay_buffer=self.buffer,
            heat=args.heat,
            fps=args.fps,
            clip_v=args.clip_v,
            n_landmark=args.landmark,
            initial_sample=args.initial_sample,
            jump_temp=args.jump_temp,
        )

        self.can_jump = False
        self.goal_loss = 0
        self.jump_ref_goal_loss = torch.zeros(1).to(self.device)
        self.weighted_multi_path_fraction = 0.0
        self.weighted_distance_span = 0.0
        self.weighted_entropy = 1.0
        self.weighted_max_weight = 0.0
        self.weighted_beta_effective = 0.0

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.actor_optim.param_groups:
            param_group["lr"] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.critic_optim.param_groups:
            param_group["lr"] = lr_critic

    def learn(self):
        self.sum_goal_loss = torch.zeros(1).to(self.device)
        self.env_timestep = 0
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions, ep_sg, ep_sg_series, ep_sg_dists, ep_path_mask = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            # Modern Gymnasium: reset returns (obs, info)
            observation, info = self.env.reset()
            obs = observation["observation"]
            ag = observation["achieved_goal"]
            g = observation["desired_goal"]

            for t in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if (
                        np.random.uniform() < self.args.plan_eps
                        and self.buffer.current_size > self.args.initial_sample
                    ):
                        action = self.planner_policy(
                            act_obs,
                            act_g,
                            self.args.plan_budget,
                            ref_loss=getattr(self, "jump_ref_goal_loss", self.goal_loss),
                            jump=self.args.jump,
                        )
                        subgoal = self.planner_policy.subgoal
                        subgoal_series = self.planner_policy.goal_series
                        subgoal_dist_series = self.planner_policy.goal_distance_series
                        path_len = self.planner_policy.path_len
                    else:
                        action = self.explore_policy(act_obs, act_g)
                        subgoal = g
                        subgoal_series = g.reshape(1, -1)
                        subgoal_series = np.repeat(subgoal_series, self.args.plan_budget, axis=0)
                        subgoal_dist_series = np.ones(self.args.plan_budget, dtype=np.float32)
                        path_len = 1
                    # feed the actions into the environment
                # Modern Gymnasium: step returns 5 values
                observation_new, reward, terminated, truncated, info = self.env.step(action)
                self.env_timestep += 1
                obs_new = observation_new["observation"]
                ag_new = observation_new["achieved_goal"]
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_sg.append(subgoal.copy())
                ep_sg_series.append(subgoal_series.copy())
                ep_sg_dists.append(subgoal_dist_series.copy())
                ep_path_mask.append(
                    np.array([1] * path_len + [0] * (self.args.plan_budget - path_len))
                )
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            mb_sg = np.array([ep_sg])
            mb_sg_series = np.array([ep_sg_series])
            mb_sg_dists = np.array([ep_sg_dists])
            mb_path_len = np.array([ep_path_mask])
            self.buffer.store_episode(
                [mb_obs, mb_ag, mb_g, mb_actions, mb_sg, mb_sg_series, mb_sg_dists, mb_path_len]
            )
            for n_batch in range(self.args.n_batches):
                actor_loss, critic_loss, goal_loss = self._update_network()
                self.sum_goal_loss += goal_loss.detach()
                if n_batch % self.args.period == 0:
                    self._soft_update_target_network(
                        self.actor_target_network, self.actor_network
                    )
                    self._soft_update_target_network(
                        self.critic_target_network, self.critic_network
                    )
            self.goal_loss = self.sum_goal_loss / self.args.n_batches
            self.sum_goal_loss = 0
            # start to do the evaluation
            if epoch % self.args.eval_freq == 0:
                success_rate = self._eval_agent()
                test_sucess_rate = self._eval_test_agent(epoch=epoch)
                test_no_plan_success_rate = self._eval_test_agent_no_plan(
                    policy=self.test_policy
                )
                print(
                    "[{}] epoch is: {}, eval success rate is: {:.3f}, {:.3f}, {:.3f}".format(
                        datetime.now(),
                        epoch,
                        success_rate,
                        test_sucess_rate,
                        test_no_plan_success_rate,
                    )
                )
                print("[weighted-debug] beta_eff={:.4f}, multi_path={:.4f}, distance_span={:.4f}, entropy={:.4f}, max_weight={:.4f}".format(
                    getattr(self, "weighted_beta_effective", 0.0),
                    getattr(self, "weighted_multi_path_fraction", 0.0),
                    getattr(self, "weighted_distance_span", 0.0),
                    getattr(self, "weighted_entropy", 1.0),
                    getattr(self, "weighted_max_weight", 0.0),
                ))
                # torch.save([self.critic_network.state_dict()], \
                #            self.model_path + '/critic_model_' +str(epoch) +'.pt')
                # torch.save([self.actor_network.state_dict()], \
                #             self.model_path + '/actor_model_' +str(epoch) +'.pt')
                # torch.save(self.buffer, self.model_path + '/replaybuffer.pt')

                self.writer.add_scalar(
                    "data/train" + self.args.env_name + self.args.metric,
                    success_rate,
                    epoch,
                )
                self.writer.add_scalar(
                    "data/test" + self.args.env_name + self.args.metric,
                    test_sucess_rate,
                    epoch,
                )
                self.writer.add_scalar(
                    "data/test_no_plan" + self.args.env_name + self.args.metric,
                    test_no_plan_success_rate,
                    epoch,
                )
                self.writer.add_scalar("data/critic_loss", critic_loss, epoch)
                self.writer.add_scalar("data/actor_loss", actor_loss, epoch)
                self.writer.add_scalar("data/goal_loss", goal_loss, epoch)
                self.writer.add_scalar(
                    "data/env_timestep", self.env_timestep, epoch
                )

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += (
            self.args.noise_eps
            * self.env_params["action_max"]
            * np.random.randn(*action.shape)
        )
        action = np.clip(
            action, -self.env_params["action_max"], self.env_params["action_max"]
        )
        # random actions...
        if np.random.randn() < self.args.random_eps:
            action = np.random.uniform(
                low=-self.env_params["action_max"],
                high=self.env_params["action_max"],
                size=self.env_params["action"],
            )
        return action

    def explore_policy(self, obs, goal):
        pi = self.actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(
            low=-self.env_params["action_max"],
            high=self.env_params["action_max"],
            size=self.env_params["action"],
        )
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        return actions

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data
                + self.args.polyak * target_param.data
            )

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = o, g
        transitions["obs_next"], transitions["g_next"] = o_next, g
        ag_next = transitions["ag_next"]

        # start to do the update
        obs_cur = transitions["obs"]
        g_cur = transitions["g"]
        obs_next = transitions["obs_next"]
        g_next = transitions["g_next"]
        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)

        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32).to(
            self.device
        )
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32).to(self.device)
        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_network(obs_next, g_next)
            q_next_value = self.critic_target_network(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.critic_target_network.gamma * q_next_value
            target_q_value = target_q_value.detach()
            clip_return = 1 / (1 - self.critic_target_network.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0.0)
        # the q loss
        real_q_value = self.critic_network(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        forward_loss = (
            self.critic_network(obs_cur, ag_next, actions_tensor).pow(2).mean()
        )
        critic_loss += forward_loss
        # the actor loss
        actions_real = self.actor_network(obs_cur, g_cur)
        actor_loss = -self.critic_network(obs_cur, g_cur, actions_real).mean()
        # start to update the network

        goal_loss = torch.zeros(1).to(self.device)
        goal_loss_uniform = torch.zeros(1).to(self.device)
        if self.buffer.current_size > self.args.initial_sample:
            # WEIGHTED-PIG-V2 START
            # The final-goal policy receives gradients; subgoal policies are stop-gradient teachers.
            actions_g = self.actor_network(obs_cur, g_cur)
            sg_series = torch.tensor(
                transitions["sg_series"], dtype=torch.float32
            ).to(self.device)
            path_mask = torch.tensor(
                transitions["path_mask"], dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                actions_sg_list = []
                for i in range(sg_series.shape[1]):
                    actions_sg_list.append(
                        self.actor_network(obs_cur, sg_series[:, i, :])
                    )
                actions_sg = torch.stack(actions_sg_list, dim=1)

            actions_g_expand = actions_g.unsqueeze(1).repeat(
                1, self.args.plan_budget, 1
            )
            per_subgoal_loss = ((actions_g_expand - actions_sg) ** 2).mean(dim=2)

            valid = path_mask > 0.0
            valid_count = path_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            uniform_prob = path_mask / valid_count
            prob_weights = uniform_prob
            beta_effective = 0.0
            distance_span_value = 0.0

            weighting = getattr(self.args, "goal_loss_weighting", "uniform")
            beta_target = float(getattr(self.args, "goal_loss_beta", 0.0))

            if weighting != "uniform" and beta_target > 0.0:
                if weighting == "graph":
                    if "sg_dists" not in transitions:
                        raise KeyError(
                            "Weighted graph loss needs transitions['sg_dists']. "
                            "Patch planner and replay buffer before training."
                        )
                    distances = torch.tensor(
                        transitions["sg_dists"], dtype=torch.float32
                    ).to(self.device)
                elif weighting == "euclidean":
                    ag_cur = torch.tensor(
                        transitions["ag"], dtype=torch.float32
                    ).to(self.device)
                    distances = torch.norm(
                        sg_series - ag_cur.unsqueeze(1), dim=2
                    )
                elif weighting == "index":
                    distances = torch.arange(
                        1,
                        self.args.plan_budget + 1,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0).repeat(path_mask.shape[0], 1)
                else:
                    raise ValueError(
                        "Unknown goal_loss_weighting: {}".format(weighting)
                    )

                # Protect the softmax from NaN/Inf and normalize each planned path.
                distances = torch.where(
                    torch.isfinite(distances),
                    distances,
                    torch.zeros_like(distances),
                )
                masked_min = distances.masked_fill(~valid, 1e9).min(
                    dim=1, keepdim=True
                )[0]
                masked_max = distances.masked_fill(~valid, -1e9).max(
                    dim=1, keepdim=True
                )[0]
                span = (masked_max - masked_min).clamp_min(1e-6)

                if getattr(
                    self.args, "goal_loss_distance_normalization", "path"
                ) == "path":
                    normalized_distances = (distances - masked_min) / span
                else:
                    normalized_distances = distances

                warmup = int(getattr(self.args, "goal_loss_warmup_steps", 0))
                ramp = max(
                    int(getattr(self.args, "goal_loss_beta_ramp_steps", 1)), 1
                )
                beta_progress = min(
                    max((float(self.env_timestep) - warmup) / float(ramp), 0.0),
                    1.0,
                )
                beta_effective = beta_target * beta_progress

                logits = -beta_effective * normalized_distances
                logits = logits.masked_fill(~valid, -1e9)
                distance_prob = torch.softmax(logits, dim=1)
                distance_prob = distance_prob * path_mask
                distance_prob = distance_prob / distance_prob.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)

                uniform_mix = float(
                    getattr(self.args, "goal_loss_uniform_mix", 0.25)
                )
                uniform_mix = min(max(uniform_mix, 0.0), 1.0)
                prob_weights = (
                    uniform_mix * uniform_prob
                    + (1.0 - uniform_mix) * distance_prob
                )
                prob_weights = prob_weights * path_mask
                prob_weights = prob_weights / prob_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                distance_span_value = float(
                    ((masked_max - masked_min) * (valid_count > 1).float())
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )

            # Critical correction:
            # multiply by valid_count and divide globally so beta=0 exactly
            # recovers the official PIG reduction, even when path lengths differ.
            relative_weights = prob_weights * valid_count
            denominator = path_mask.sum().clamp_min(1.0)
            goal_loss = (
                relative_weights * per_subgoal_loss * path_mask
            ).sum() / denominator
            goal_loss_uniform = (
                per_subgoal_loss * path_mask
            ).sum() / denominator

            actor_loss += goal_loss * self.args.lambda_goal_loss

            # Keep subgoal skipping calibrated to the original/uniform PIG loss.
            self.jump_ref_goal_loss = goal_loss_uniform.detach()

            with torch.no_grad():
                entropy = -(
                    prob_weights.clamp_min(1e-8)
                    * torch.log(prob_weights.clamp_min(1e-8))
                    * path_mask
                ).sum(dim=1)
                entropy_den = torch.log(valid_count.squeeze(1).clamp_min(2.0))
                multi = valid_count.squeeze(1) > 1.0
                if multi.any():
                    normalized_entropy = (
                        entropy[multi] / entropy_den[multi]
                    ).mean()
                else:
                    normalized_entropy = torch.ones(1).to(self.device)

                self.weighted_multi_path_fraction = float(
                    multi.float().mean().cpu().item()
                )
                self.weighted_distance_span = distance_span_value
                self.weighted_entropy = float(
                    normalized_entropy.detach().cpu().item()
                )
                self.weighted_max_weight = float(
                    prob_weights.max(dim=1)[0].mean().detach().cpu().item()
                )
                self.weighted_beta_effective = float(beta_effective)
            # WEIGHTED-PIG-V2 END
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return actor_loss, critic_loss, goal_loss

    # do the evaluation
    def _eval_agent(self, policy=None):
        if policy is None:
            policy = self.test_policy

        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            # Modern Gymnasium: reset returns (obs, info)
            observation, info = self.env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]
            for _ in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(act_obs, act_g)
                # Modern Gymnasium: step returns 5 values
                observation_new, reward, terminated, truncated, info = self.env.step(actions)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                per_success_rate.append(info["is_success"])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate

    def _eval_test_agent(self, policy=None, epoch=0):
        if policy is None:
            policy = self.planner_policy
            policy.reset()

        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            policy.reset()
            per_success_rate = []
            # Modern Gymnasium: reset returns (obs, info)
            observation, info = self.test_env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]

            for num in range(self.env_params["max_test_timesteps"]):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(
                        act_obs,
                        act_g,
                        self.args.plan_budget,
                        ref_loss=getattr(self, "jump_ref_goal_loss", self.goal_loss),
                        jump=self.args.jump,
                    )
                # Modern Gymnasium: step returns 5 values
                observation_new, reward, terminated, truncated, info = self.test_env.step(actions)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                per_success_rate.append(info["is_success"])
            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate

    def _eval_test_agent_no_plan(self, policy):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            # policy.reset()
            per_success_rate = []
            # Modern Gymnasium: reset returns (obs, info)
            observation, info = self.test_env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]
            for num in range(self.env_params["max_test_timesteps"]):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(act_obs, act_g)
                # Modern Gymnasium: step returns 5 values
                observation_new, reward, terminated, truncated, info = self.test_env.step(actions)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                per_success_rate.append(info["is_success"])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate

    def pairwise_value(self, obs, goal):
        assert obs.shape[0] == goal.shape[0]
        actions = self.actor_network(obs, goal)
        dist = self.critic_network.base(obs, goal, actions).squeeze(-1)
        return -dist