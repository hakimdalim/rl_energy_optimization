# rl_energy_optimization
# RL Energy Cost Optimization System

## Overview

The RL Energy Cost Optimization System is a comprehensive reinforcement learning framework designed to optimize energy system operations and minimize operational costs. The system uses Deep Q-Network (DQN) agents to learn optimal control strategies for gas plants, energy storage systems, hydrogen production, and demand response programs.

## Key Features

- **Multi-objective Cost Optimization**: Minimizes total operational costs including gas generation, electricity purchases, storage operations, and demand response
- **Real-time Decision Making**: Hourly operational decisions based on market conditions and system state
- **Renewable Energy Integration**: Optimizes renewable energy utilization and storage scheduling
- **Hydrogen Production Optimization**: Strategic hydrogen production during favorable market conditions
- **Comprehensive Analytics**: Detailed performance analysis and visualization tools
- **SMARD Data Integration**: Native support for German energy market data

## System Architecture

### Core Components

1. **Energy System Simulator**: Models gas plants, storage systems, and hydrogen production
2. **Cost Calculator**: Implements multi-component cost function with operational constraints
3. **RL Agent**: DQN-based learning agent for optimal policy discovery
4. **Data Processor**: Handles SMARD data preprocessing and feature engineering
5. **Analysis Engine**: Comprehensive result analysis and visualization

### Cost Function

The system optimizes the following cost equation:

```
Total_Cost = C_gas + C_ele + C_dr + C_ope + C_wp - C_h2
```

Where:
- `C_gas`: Natural gas generation costs
- `C_ele`: Electricity purchase costs
- `C_dr`: Demand response costs
- `C_ope`: Operational maintenance costs
- `C_wp`: Storage pumping costs
- `C_h2`: Hydrogen production revenue

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Minimum 8GB RAM
- 2GB available disk space for results storage

### Dependencies

```bash
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
```

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd rl-energy-optimization
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
rl-energy-optimization/
├── energy_cost_rl.py              # Core RL framework
├── run_smard_optimization.py      # Data processing and execution
├── save_analysis_results.py       # Results export and visualization
├── complete_rl_analysis.py        # End-to-end analysis pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── preprocessed_smard_data/       # Input data directory
    └── *.pkl                      # Preprocessed SMARD data files
```

## Quick Start

### Basic Usage

1. **Prepare Data**: Place your preprocessed SMARD data in the `preprocessed_smard_data/` directory
2. **Run Analysis**: Execute the complete analysis pipeline
3. **Review Results**: Check the generated results directory

```bash
python complete_rl_analysis.py
```

### Advanced Usage

For custom configurations and parameters:

```bash
python run_smard_optimization.py
```

## Configuration

### System Parameters

The energy system configuration can be customized in `create_optimized_config_for_smard()`:

```python
config = EnergySystemConfig(
    gas_capacity=100.0,              # Gas plant capacity (MWh)
    storage_capacity=50.0,           # Battery storage capacity (MWh)
    h2_production_capacity=20.0,     # Electrolyzer capacity (MWh)
    gas_base_cost=45.0,              # Gas cost (EUR/MWh)
    h2_selling_price=80.0,           # H2 market price (EUR/MWh)
    storage_efficiency=0.85,         # Round-trip efficiency
    gas_startup_cost=500.0,          # Gas plant startup cost (EUR)
    storage_maintenance_cost=2.0,    # Storage maintenance (EUR/MWh)
    dr_cost_factor=1.2               # Demand response premium factor
)
```

### Training Parameters

```python
training_params = {
    'episodes': 100,                 # Number of training episodes
    'learning_rate': 0.001,          # Agent learning rate
    'epsilon_decay': 0.995,          # Exploration decay rate
    'memory_size': 2000,             # Experience replay buffer size
    'batch_size': 32                 # Training batch size
}
```

## Data Input Format

### SMARD Data Requirements

The system expects preprocessed SMARD data in pickle format with the following structure:

```python
# Dictionary format
{
    'data': DataFrame,               # Main energy data
    'metadata': dict,                # Data information
    'config': dict                   # Processing configuration
}
```

### Required Data Columns

- `price_eur_mwh`: Electricity prices (EUR/MWh)
- `wind_mwh`: Wind generation (MWh)
- `pv_mwh`: Solar generation (MWh)
- `load_mwh`: Electricity demand (MWh)
- `gas_mwh`: Gas generation (MWh)
- `hour`: Hour of day (0-23)

## Results and Output

### Output Directory Structure

```
rl_optimization_results/
└── run_YYYYMMDD_HHMMSS/
    ├── metadata/
    │   └── run_metadata.json       # Complete run configuration
    ├── data/
    │   ├── training_history.csv    # Episode-by-episode training data
    │   ├── training_results.json   # Training performance metrics
    │   └── performance_results.json # Final performance comparison
    ├── models/
    │   ├── trained_agent.pkl       # Saved RL agent model
    │   └── system_config.pkl       # System configuration
    ├── visualizations/
    │   ├── training/
    │   │   ├── training_overview.png
    │   │   └── learning_analysis.png
    │   ├── performance/
    │   │   └── performance_overview.png
    │   └── analysis/
    │       └── data_analysis.png
    └── reports/
        └── summary_report.md       # Executive summary
```

### Key Performance Metrics

- **Cost Reduction Percentage**: Improvement over baseline strategy
- **Total Cost Savings**: Absolute savings in EUR
- **Gas Utilization Frequency**: Percentage of time gas plant operates
- **Storage Activity**: Battery charging/discharging frequency
- **Hydrogen Production Frequency**: H2 production operational time

## Performance Analysis

### Training Metrics

- **Episode Costs**: Cost progression during training
- **Learning Curve**: Improvement rate over episodes
- **Convergence Analysis**: Training stability assessment
- **Exploration vs Exploitation**: Agent behavior analysis

### Operational Metrics

- **Cost Breakdown**: Detailed cost component analysis
- **Action Frequency**: Operational decision patterns
- **Golden Hour Utilization**: Low-price period exploitation
- **Renewable Integration**: Renewable energy utilization efficiency

## Troubleshooting

### Common Issues

#### Data Loading Errors

```bash
# Check pickle file structure
python -c "
import pickle
with open('path/to/file.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Type: {type(data)}')
if isinstance(data, dict):
    print(f'Keys: {list(data.keys())}')
"
```

#### Memory Issues

- Reduce training episodes for systems with limited RAM
- Decrease experience replay buffer size
- Use data sampling for large datasets

#### Convergence Problems

- Increase training episodes (recommended: 100-200)
- Adjust learning rate (try 0.01 for faster learning)
- Modify epsilon decay rate for exploration balance

### Performance Optimization

#### For Better Results

1. **Increase Training Episodes**: 100-200 episodes for production use
2. **Tune Hyperparameters**: Adjust learning rate and exploration parameters
3. **System Configuration**: Match parameters to actual energy system specifications
4. **Data Quality**: Ensure complete and accurate input data

#### For Faster Execution

1. **Reduce Episodes**: Use 20-50 episodes for testing
2. **Smaller Batch Size**: Reduce memory usage
3. **Simplified Features**: Use minimal feature sets for quick analysis

## Advanced Features

### Custom Cost Functions

Implement domain-specific cost functions by extending the `CostCalculator` class:

```python
class CustomCostCalculator(CostCalculator):
    def calculate_costs(self, state, actions, prev_gas_output=0):
        # Custom cost calculation logic
        return costs
```

### Multi-Agent Systems

The framework supports multi-agent configurations for complex energy systems:

```python
# Multiple agents for different system components
agents = {
    'gas_plant': SimpleDQNAgent(),
    'storage': SimpleDQNAgent(),
    'h2_production': SimpleDQNAgent()
}
```

### Real-time Integration

For production deployment, implement real-time data feeds:

```python
class RealTimeDataFeed:
    def get_current_state(self):
        # Fetch real-time market data
        return current_state
```

## API Reference

### Core Classes

#### `EnergySystemConfig`
Configuration container for energy system parameters.

#### `EnergyState`
Represents the current state of the energy system including price, renewable generation, load, and storage level.

#### `SimpleDQNAgent`
Deep Q-Network agent for learning optimal control policies.

#### `EnergyRLEnvironment`
Reinforcement learning environment for energy system simulation.

#### `CostCalculator`
Calculates operational costs based on system actions and market conditions.

### Key Functions

#### `train_energy_rl_agent(data, episodes=100)`
Trains the RL agent on provided energy data.

#### `run_smard_rl_optimization(pickle_path, episodes=100)`
Complete optimization workflow for SMARD data.

#### `save_complete_analysis(agent, costs, rewards, test_results, config, data)`
Saves comprehensive analysis results and generates visualizations.

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Submit pull request with detailed description

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation for API changes

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For technical support and questions:

- Create an issue in the repository
- Provide system specifications and error logs
- Include minimal reproducible examples

## Changelog

### Version 1.0.0
- Initial release with core RL optimization framework
- SMARD data integration
- Comprehensive analysis and visualization tools
- Multi-component cost optimization

### Version 1.1.0
- Enhanced visualization capabilities
- Improved data preprocessing
- Performance optimization
- Extended configuration options

## Acknowledgments

- German Federal Network Agency (Bundesnetzagentur) for SMARD data
- Open-source reinforcement learning community
- Energy system modeling research contributions

## Citation

If you use this software in your research, please cite:

```bibtex
@software{rl_energy_optimization,
  title={RL Energy Cost Optimization System},
  author={Hakim Dalim},
  year={2024},
  url={https://github.com/hakimdalim/rl-energy-optimization}
}
```
