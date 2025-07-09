import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

@dataclass
class EnergySystemConfig:
    """Configuration for the energy system components"""
    # System capacities (MWh)
    gas_capacity: float = 100.0
    storage_capacity: float = 50.0
    h2_production_capacity: float = 20.0
    
    # Cost parameters (€/MWh or €/MW)
    gas_base_cost: float = 45.0  # €/MWh
    storage_efficiency: float = 0.85  # Round-trip efficiency
    h2_production_efficiency: float = 0.6  # Electricity to H2
    h2_selling_price: float = 80.0  # €/MWh equivalent
    
    # Operational costs
    gas_startup_cost: float = 500.0  # €
    storage_maintenance_cost: float = 2.0  # €/MWh
    dr_cost_factor: float = 1.2  # Demand response premium

class EnergyState:
    """Represents the current state of the energy system"""
    def __init__(self, price: float, renewable: float, load: float, 
                 hour: int, storage_level: float = 0.5):
        self.price = price  # €/MWh
        self.renewable = renewable  # MWh available
        self.load = load  # MWh demand
        self.hour = hour  # Hour of day (0-23)
        self.storage_level = storage_level  # Fraction of capacity (0-1)
        
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL agent"""
        return np.array([
            self.price / 100.0,  # Normalize price
            self.renewable,
            self.load,
            self.hour / 23.0,  # Normalize hour
            self.storage_level
        ])

class EnergyActions:
    """Defines possible actions for the RL agent"""
    def __init__(self, config: EnergySystemConfig):
        self.config = config
        # Action space: [gas_output, storage_action, h2_production, dr_activation]
        # Values normalized between 0 and 1
        
    def decode_action(self, action: np.ndarray) -> Dict[str, float]:
        """Convert normalized action to actual system commands"""
        return {
            'gas_output': action[0] * self.config.gas_capacity,  # MWh
            'storage_action': (action[1] - 0.5) * 2,  # -1 (discharge) to +1 (charge)
            'h2_production': action[2] * self.config.h2_production_capacity,  # MWh
            'dr_activation': action[3]  # 0 to 1 (fraction of load to shift)
        }

class CostCalculator:
    """Calculates the total cost based on system actions"""
    def __init__(self, config: EnergySystemConfig):
        self.config = config
        
    def calculate_costs(self, state: EnergyState, actions: Dict[str, float], 
                       prev_gas_output: float = 0) -> Dict[str, float]:
        """
        Calculate all cost components:
        Total_cost = C_gas + C_ele + C_dr + C_ope + C_wp - C_h2
        """
        costs = {}
        
        # C_gas: Natural gas costs
        gas_output = actions['gas_output']
        costs['C_gas'] = gas_output * self.config.gas_base_cost
        
        # Add startup costs if gas plant was off and now starting
        if prev_gas_output < 0.1 and gas_output > 0.1:
            costs['C_gas'] += self.config.gas_startup_cost
            
        # C_ele: Electricity purchase costs (when renewable + gas < load)
        renewable_available = min(state.renewable, state.load)
        total_generation = renewable_available + gas_output
        
        # Account for storage discharge
        storage_discharge = max(0, -actions['storage_action'] * self.config.storage_capacity)
        total_generation += storage_discharge * self.config.storage_efficiency
        
        electricity_needed = max(0, state.load - total_generation)
        costs['C_ele'] = electricity_needed * max(0, state.price)
        
        # C_dr: Demand response costs
        dr_load = actions['dr_activation'] * state.load
        costs['C_dr'] = dr_load * state.price * self.config.dr_cost_factor
        
        # C_ope: Operational costs (storage maintenance)
        storage_usage = abs(actions['storage_action']) * self.config.storage_capacity
        costs['C_ope'] = storage_usage * self.config.storage_maintenance_cost
        
        # C_wp: Storage charging costs (pumping costs)
        storage_charge = max(0, actions['storage_action'] * self.config.storage_capacity)
        costs['C_wp'] = storage_charge * max(0, state.price)
        
        # C_h2: Hydrogen production revenue (negative cost)
        h2_production = actions['h2_production']
        costs['C_h2'] = -h2_production * self.config.h2_selling_price
        
        # Total cost
        costs['total'] = sum(costs.values())
        
        return costs

class SimpleDQNAgent:
    """Simplified Deep Q-Network agent for energy cost optimization"""
    def __init__(self, state_size: int = 5, action_size: int = 4, 
                 learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Simple Q-table for demonstration (replace with neural network for real use)
        self.q_table = np.random.normal(0, 0.1, (1000, action_size))
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def state_to_index(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index for Q-table"""
        # Simple discretization (improve this for better performance)
        discretized = (state * 10).astype(int)
        return hash(tuple(discretized)) % len(self.q_table)
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Random action (exploration)
            return np.random.random(self.action_size)
        
        # Greedy action (exploitation)
        state_idx = self.state_to_index(state)
        q_values = self.q_table[state_idx]
        # Convert Q-values to action probabilities
        action = np.random.random(self.action_size)
        best_action_idx = np.argmax(q_values)
        action[best_action_idx] = max(action[best_action_idx], 0.7)
        return np.clip(action, 0, 1)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_idx = self.state_to_index(next_state)
                target += 0.95 * np.max(self.q_table[next_state_idx])
            
            state_idx = self.state_to_index(state)
            # Simple Q-learning update
            action_idx = np.argmax(action)  # Simplified
            self.q_table[state_idx][action_idx] += self.learning_rate * (
                target - self.q_table[state_idx][action_idx]
            )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class EnergyRLEnvironment:
    """RL Environment for energy cost optimization"""
    def __init__(self, data: pd.DataFrame, config: EnergySystemConfig):
        self.data = data.copy()
        self.config = config
        self.cost_calculator = CostCalculator(config)
        self.reset()
        
    def reset(self) -> EnergyState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.storage_level = 0.5  # Start at 50% storage
        self.prev_gas_output = 0.0
        self.total_cost = 0.0
        self.cost_history = []
        
        return self._get_current_state()
    
    def _get_current_state(self) -> EnergyState:
        """Get current state from data"""
        if self.current_step >= len(self.data):
            self.current_step = 0
            
        row = self.data.iloc[self.current_step]
        
        # Extract state variables (adjust column names based on your data)
        price = row.get('price_eur_mwh', 0)
        renewable = row.get('wind_mwh', 0) + row.get('pv_mwh', 0)
        load = row.get('load_mwh', 0)
        hour = row.get('hour', self.current_step % 24)
        
        return EnergyState(price, renewable, load, hour, self.storage_level)
    
    def step(self, action: np.ndarray) -> Tuple[EnergyState, float, bool, Dict]:
        """Execute one step in the environment"""
        # Get current state
        state = self._get_current_state()
        
        # Decode action
        actions = EnergyActions(self.config).decode_action(action)
        
        # Calculate costs
        costs = self.cost_calculator.calculate_costs(
            state, actions, self.prev_gas_output
        )
        
        # Update storage level
        storage_change = actions['storage_action'] * self.config.storage_capacity
        self.storage_level = np.clip(
            self.storage_level + storage_change / self.config.storage_capacity,
            0, 1
        )
        
        # Update state
        self.prev_gas_output = actions['gas_output']
        self.total_cost += costs['total']
        self.cost_history.append(costs)
        
        # Calculate reward (negative cost)
        reward = -costs['total'] / 1000.0  # Scale reward
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        # Get next state
        next_state = self._get_current_state() if not done else state
        
        info = {
            'costs': costs,
            'actions': actions,
            'storage_level': self.storage_level
        }
        
        return next_state, reward, done, info

def train_energy_rl_agent(data: pd.DataFrame, episodes: int = 100) -> SimpleDQNAgent:
    """Train the RL agent on energy data"""
    config = EnergySystemConfig()
    env = EnergyRLEnvironment(data, config)
    agent = SimpleDQNAgent()
    
    episode_rewards = []
    episode_costs = []
    
    print("Training Energy Cost Optimization Agent...")
    print("=" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_cost = 0
        
        while True:
            # Agent chooses action
            action = agent.act(state.to_array())
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(
                state.to_array(), action, reward, 
                next_state.to_array(), done
            )
            
            total_reward += reward
            total_cost += info['costs']['total']
            
            state = next_state
            
            if done:
                break
        
        # Train agent
        agent.replay()
        
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        
        if episode % 10 == 0:
            avg_cost = np.mean(episode_costs[-10:])
            print(f"Episode {episode:3d} | Avg Cost (last 10): €{avg_cost:,.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, episode_costs, episode_rewards

def analyze_rl_performance(costs: List[float], rewards: List[float]):
    """Analyze and visualize RL training performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cost reduction over episodes
    ax1.plot(costs, alpha=0.7, label='Episode Cost')
    ax1.plot(pd.Series(costs).rolling(10).mean(), 
             label='10-Episode Average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Cost (€)')
    ax1.set_title('Cost Reduction During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot reward improvement
    ax2.plot(rewards, alpha=0.7, label='Episode Reward')
    ax2.plot(pd.Series(rewards).rolling(10).mean(), 
             label='10-Episode Average', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Reward Improvement During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
def test_with_smard_data():
    """Test the RL framework with SMARD-like data"""
    # Create sample data similar to your SMARD dataset
    np.random.seed(42)
    n_hours = 57  # Match your dataset size
    
    sample_data = pd.DataFrame({
        'price_eur_mwh': np.random.normal(50, 30, n_hours),  # €/MWh
        'wind_mwh': np.random.exponential(20, n_hours),      # Wind generation
        'pv_mwh': np.random.exponential(15, n_hours),        # Solar generation  
        'load_mwh': np.random.normal(60, 10, n_hours),       # Electricity demand
        'hour': [i % 24 for i in range(n_hours)]             # Hour of day
    })
    
    # Add some golden hours (low prices) around hour 14
    for i in range(len(sample_data)):
        if sample_data.iloc[i]['hour'] in [13, 14, 15]:
            sample_data.iloc[i, sample_data.columns.get_loc('price_eur_mwh')] *= 0.3
    
    print("Sample SMARD-like data created:")
    print(sample_data.head())
    print(f"\nData shape: {sample_data.shape}")
    print(f"Price range: €{sample_data['price_eur_mwh'].min():.2f} - €{sample_data['price_eur_mwh'].max():.2f}/MWh")
    
    # Train the agent
    agent, costs, rewards = train_energy_rl_agent(sample_data, episodes=50)
    
    # Analyze performance
    analyze_rl_performance(costs, rewards)
    
    print(f"\nTraining completed!")
    print(f"Initial average cost: €{np.mean(costs[:5]):,.2f}")
    print(f"Final average cost: €{np.mean(costs[-5:]):,.2f}")
    print(f"Cost reduction: {(1 - np.mean(costs[-5:]) / np.mean(costs[:5])) * 100:.1f}%")

if __name__ == "__main__":
    test_with_smard_data()