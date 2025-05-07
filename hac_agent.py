import numpy as np
from layer import Layer

class HACAgent:
    def __init__(self, state_dim, action_dim, goal_dim, max_action, time_limit):
        self.num_layers = 2  # Two-level hierarchy: High-level and Low-level
        self.time_limit = time_limit
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        self.layers = []
        for layer_idx in range(self.num_layers):
            layer = Layer(
                state_dim=state_dim,
                action_dim=action_dim if layer_idx == 0 else goal_dim,  # Bottom layer uses primitive actions
                goal_dim=goal_dim,
                max_action=max_action,
                layer_number=layer_idx,
            )
            self.layers.append(layer)

    def calculate_stability_reward(self, state):
        # Extract relevant information from state based on actual observation indices
        z_position = state[0]         # z-coordinate of the torso center
        orientation = state[1:5]      # w, x, y, z orientation quaternion
        vel_x, vel_y, vel_z = state[13:16]  # x, y, z velocity components
        angular_velocity = state[16:19]  # angular velocity components
        

        # Base stability reward
        stability_reward = 0.0
        
        # 1. Height penalty - gradual penalty for getting too low
        if z_position < 0.25:
            stability_reward -= (0.25 - z_position) * 5.0  # Less strong penalty
        
        # 2. Orientation penalty - reduced penalty
        quat_w = orientation[0]
        orientation_penalty = 0.2 * (1.0 - quat_w**2)  # Reduced from 0.5 to 0.2
        stability_reward -= orientation_penalty
        
        # 3. Forward motion reward - increased weight
        forward_reward_weight = 2.0  # Increased from 1.0 to 2.0
        forward_velocity = (vel_x**2 + vel_y**2)**0.5
        forward_reward = forward_reward_weight * forward_velocity
        
        # 4. Control cost - reduced weight
        ctrl_cost_weight = 0.05  # Reduced from 0.1 to 0.05
        ctrl_cost = ctrl_cost_weight * np.sum(np.square(angular_velocity))
        
        # 5. Vertical velocity penalty - reduced
        z_vel_penalty = 0.2 * abs(vel_z)  # Reduced from 0.5 to 0.2
        
        # Combined reward
        combined_reward = stability_reward + forward_reward - ctrl_cost - z_vel_penalty
        
        return combined_reward

    def run_episode(self, env, episode_idx, exploration_scale=0.3):
        state, info = env.reset()
        goal = state['desired_goal']
        obs = state['observation']
        episode_reward = 0
        
        # Set exploration scale for all layers
        for layer in self.layers:
            layer.exploration_noise = layer.base_exploration_noise * exploration_scale
        
        # Track the latest observation across layers
        self.latest_observation = obs
        
        # Run the top layer for the full episode
        success = self._run_layer(
            layer_idx=self.num_layers - 1,
            state=obs,
            goal=goal,
            env=env,
            episode_reward=episode_reward
        )

        return success

    def _run_layer(self, layer_idx, state, goal, env, episode_reward):
        layer = self.layers[layer_idx]
        current_state = state
        timesteps = 0
        max_horizon = self.time_limit if layer_idx == 0 else self.time_limit // 2
        
        while timesteps < max_horizon:
            if layer_idx == 0:  # Low-level layer (primitive actions)
                # Execute primitive action
                action = layer.select_action(current_state, goal)
                next_state_dict, reward, terminated, truncated, info = env.step(action)
                
                next_state = next_state_dict["observation"]
                # Update our cached latest observation
                self.latest_observation = next_state
                
                # Use a weighted combination of environment reward and stability reward
                stability_reward = self.calculate_stability_reward(next_state)
                
                # The environment reward already includes forward motion and control costs
                # We'll add our stability components without double-counting
                env_reward_weight = 2.0  # Give higher weight to env reward
                stability_weight = 0.5   # But still include stability considerations
                
                adjusted_reward = (env_reward_weight * reward) + (stability_weight * stability_reward)
                episode_reward += adjusted_reward
                
                # Store transition with hindsight
                self._store_transition_with_hindsight(
                    layer, current_state, action, next_state, 
                    adjusted_reward, terminated or truncated, goal
                )
                
                current_state = next_state
                if terminated or truncated:
                    return False
             
                    
            else:  # High-level layer (subgoals)
                # Generate subgoal
                subgoal = layer.select_action(current_state, goal)
                
                # Run lower layer to achieve subgoal
                subgoal_achieved = self._run_layer(
                    layer_idx - 1,
                    current_state,
                    subgoal,
                    env,
                    episode_reward
                )
                # Check if environment terminated during lower layer execution
                if not subgoal_achieved and hasattr(env, 'terminated') and env.terminated:
                    return False  # Propagate termination upward
                
                # Use the latest observation that was updated by the lower layer's steps
                next_state = self.latest_observation
                
                # Evaluate if subgoal was achieved
                subgoal_distance = np.linalg.norm(next_state[:self.goal_dim] - subgoal)
                subgoal_reward = 1.0 if subgoal_distance < 0.5 else -0.1
                
                # Store high-level transition with hindsight
                self._store_transition_with_hindsight(
                    layer, current_state, subgoal, next_state, 
                    subgoal_reward, False, goal
                )
                
                current_state = next_state
                
            timesteps += 1
                
        # Success is reaching close to the goal
        goal_distance = np.linalg.norm(current_state[:self.goal_dim] - goal)
        return goal_distance < 1.0  # Adjust threshold as needed
        
    def _store_transition_with_hindsight(self, layer, state, action, next_state, reward, done, goal):
        # Store original transition
        layer.replay_buffer.add(state, action, next_state, reward, done, goal)
        
        # Extract achieved goal from next_state (typically x,y position for Ant)
        achieved_goal = next_state[:self.goal_dim]
        
        # Compute distance between achieved goal and intended goal
        goal_distance = np.linalg.norm(achieved_goal - goal)
        
        # Compute hindsight reward - success if close enough to achieved goal
        hindsight_reward = 1.0 if goal_distance < 0.1 else -0.1
        
        # Store hindsight transition with the achieved goal as the intended goal
        layer.replay_buffer.add(state, action, next_state, hindsight_reward, done, achieved_goal)

    def train(self, batch_size):
        for layer in self.layers:
            if layer.replay_buffer.size >= batch_size:
                layer.train(batch_size)