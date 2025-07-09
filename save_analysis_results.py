# save_analysis_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AnalysisResultsSaver:
    """Comprehensive saver for RL optimization results and visualizations"""
    
    def __init__(self, base_dir: str = "./rl_optimization_results"):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"run_{self.timestamp}"
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized directory structure"""
        directories = [
            self.run_dir,
            self.run_dir / "metadata",
            self.run_dir / "visualizations",
            self.run_dir / "visualizations" / "training",
            self.run_dir / "visualizations" / "performance",
            self.run_dir / "visualizations" / "analysis",
            self.run_dir / "data",
            self.run_dir / "models",
            self.run_dir / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Created results directory: {self.run_dir}")
    
    def save_training_metadata(self, config, training_params, data_info):
        """Save all training configuration and metadata"""
        
        metadata = {
            "run_info": {
                "timestamp": self.timestamp,
                "run_directory": str(self.run_dir),
                "training_completed": datetime.now().isoformat()
            },
            "data_info": {
                "source_file": data_info.get("source_file", ""),
                "data_shape": data_info.get("shape", [0, 0]),
                "date_range": data_info.get("date_range", {}),
                "columns": data_info.get("columns", []),
                "golden_hours_count": data_info.get("golden_hours", 0)
            },
            "system_config": {
                "gas_capacity": config.gas_capacity,
                "storage_capacity": config.storage_capacity,
                "h2_production_capacity": config.h2_production_capacity,
                "gas_base_cost": config.gas_base_cost,
                "storage_efficiency": config.storage_efficiency,
                "h2_production_efficiency": config.h2_production_efficiency,
                "h2_selling_price": config.h2_selling_price,
                "gas_startup_cost": config.gas_startup_cost,
                "storage_maintenance_cost": config.storage_maintenance_cost,
                "dr_cost_factor": config.dr_cost_factor
            },
            "training_params": training_params
        }
        
        # Save metadata as JSON
        metadata_file = self.run_dir / "metadata" / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"💾 Saved metadata: {metadata_file}")
        return metadata
    
    def save_training_results(self, agent, costs, rewards, test_results):
        """Save all training results and performance metrics"""
        
        # Training performance data
        training_data = {
            "episode_costs": costs,
            "episode_rewards": rewards,
            "episodes_count": len(costs),
            "final_cost": costs[-1] if costs else 0,
            "initial_cost": costs[0] if costs else 0,
            "best_cost": min(costs) if costs else 0,
            "worst_cost": max(costs) if costs else 0,
            "cost_improvement": ((costs[0] - costs[-1]) / costs[0] * 100) if costs and costs[0] != 0 else 0
        }
        
        # Test results
        performance_data = {
            "rl_agent_cost": test_results.get("rl_cost", 0),
            "baseline_cost": test_results.get("baseline_cost", 0),
            "cost_improvement_percent": test_results.get("improvement_percent", 0),
            "total_cost_savings": test_results.get("cost_savings", 0),
            "actions_summary": test_results.get("actions_summary", {})
        }
        
        # Save training data
        training_file = self.run_dir / "data" / "training_results.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Save performance data
        performance_file = self.run_dir / "data" / "performance_results.json"
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Save training arrays as CSV for easy analysis
        training_df = pd.DataFrame({
            'episode': range(len(costs)),
            'cost': costs,
            'reward': rewards
        })
        training_df.to_csv(self.run_dir / "data" / "training_history.csv", index=False)
        
        print(f"💾 Saved training results: {training_file}")
        print(f"💾 Saved performance data: {performance_file}")
        print(f"💾 Saved training history: {self.run_dir / 'data' / 'training_history.csv'}")
        
        return training_data, performance_data
    
    def save_model_artifacts(self, agent, config):
        """Save the trained model and configuration"""
        
        # Save the trained agent
        agent_file = self.run_dir / "models" / "trained_agent.pkl"
        with open(agent_file, 'wb') as f:
            pickle.dump(agent, f)
        
        # Save configuration
        config_file = self.run_dir / "models" / "system_config.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"🤖 Saved trained agent: {agent_file}")
        print(f"⚙️  Saved configuration: {config_file}")
    
    def create_training_visualizations(self, costs, rewards):
        """Create and save training performance visualizations"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Training Progress Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RL Training Performance Analysis', fontsize=16, fontweight='bold')
        
        # Cost reduction over episodes
        episodes = range(len(costs))
        ax1.plot(episodes, costs, alpha=0.7, linewidth=1, label='Episode Cost')
        ax1.plot(episodes, pd.Series(costs).rolling(10, min_periods=1).mean(), 
                linewidth=3, label='10-Episode Moving Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Cost (€)')
        ax1.set_title('Cost Reduction During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Reward improvement
        ax2.plot(episodes, rewards, alpha=0.7, linewidth=1, label='Episode Reward')
        ax2.plot(episodes, pd.Series(rewards).rolling(10, min_periods=1).mean(), 
                linewidth=3, label='10-Episode Moving Average')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Reward Improvement During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cost distribution
        ax3.hist(costs, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(costs), color='red', linestyle='--', linewidth=2, label=f'Mean: €{np.mean(costs):,.0f}')
        ax3.axvline(np.median(costs), color='green', linestyle='--', linewidth=2, label=f'Median: €{np.median(costs):,.0f}')
        ax3.set_xlabel('Episode Cost (€)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Cost Distribution Across Episodes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning curve (improvement over time)
        initial_avg = np.mean(costs[:5]) if len(costs) >= 5 else costs[0]
        improvement = [(initial_avg - cost) / initial_avg * 100 for cost in costs]
        ax4.plot(episodes, improvement, linewidth=2, color='green')
        ax4.fill_between(episodes, improvement, alpha=0.3, color='green')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Cumulative Cost Improvement')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        training_overview = self.run_dir / "visualizations" / "training" / "training_overview.png"
        plt.savefig(training_overview, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Learning Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Detailed Learning Analysis', fontsize=16, fontweight='bold')
        
        # Episode-by-episode improvement
        episode_improvements = []
        window = 10
        for i in range(len(costs) - window):
            early_avg = np.mean(costs[i:i+window])
            later_avg = np.mean(costs[i+window:i+2*window]) if i+2*window <= len(costs) else np.mean(costs[i+window:])
            improvement = (early_avg - later_avg) / early_avg * 100 if early_avg != 0 else 0
            episode_improvements.append(improvement)
        
        ax1.plot(range(len(episode_improvements)), episode_improvements, linewidth=2)
        ax1.set_xlabel(f'Episode (comparing {window}-episode windows)')
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title(f'Rolling {window}-Episode Improvement Rate')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        # Cost vs Reward correlation
        ax2.scatter(costs, rewards, alpha=0.6, s=30)
        ax2.set_xlabel('Episode Cost (€)')
        ax2.set_ylabel('Episode Reward')
        ax2.set_title('Cost vs Reward Relationship')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(costs, rewards, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(costs), p(sorted(costs)), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        learning_analysis = self.run_dir / "visualizations" / "training" / "learning_analysis.png"
        plt.savefig(learning_analysis, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training visualizations:")
        print(f"  - {training_overview}")
        print(f"  - {learning_analysis}")
    
    def create_performance_visualizations(self, test_results, data):
        """Create performance analysis visualizations"""
        
        # 1. Cost Breakdown Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RL Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # Cost comparison
        costs = [test_results['baseline_cost'], test_results['rl_cost']]
        labels = ['Baseline Strategy', 'RL Agent']
        colors = ['lightcoral', 'lightgreen']
        
        bars = ax1.bar(labels, costs, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Total Cost (€)')
        ax1.set_title('Total Cost Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'€{cost:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Savings visualization - handle negative savings
        savings = test_results['cost_savings']
        improvement_pct = test_results['improvement_percent']
        
        if savings > 0:
            # Positive savings - show pie chart
            ax2.pie([savings, test_results['rl_cost']], 
                   labels=[f'Savings\n€{savings:,.0f}', f'Final Cost\n€{test_results["rl_cost"]:,.0f}'],
                   colors=['gold', 'lightgreen'], autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Cost Savings: {improvement_pct:.1f}% Improvement')
        else:
            # Negative savings - show bar chart instead
            costs = [test_results['baseline_cost'], test_results['rl_cost'], abs(savings)]
            labels = ['Baseline', 'RL Agent', 'Extra Cost']
            colors = ['lightblue', 'lightcoral', 'red']
            
            bars = ax2.bar(labels, costs, color=colors, alpha=0.8)
            ax2.set_ylabel('Cost (€)')
            ax2.set_title(f'Cost Comparison: {improvement_pct:.1f}% Change')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'€{cost:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Action analysis
        actions = test_results['actions_summary']
        action_metrics = {
            'Gas Utilization': actions.get('gas_utilization', 0) * 100,
            'Storage Charging': actions.get('storage_charging_freq', 0) * 100,
            'H2 Production': actions.get('h2_production_freq', 0) * 100
        }
        
        ax3.bar(action_metrics.keys(), action_metrics.values(), 
               color=['orange', 'blue', 'green'], alpha=0.7)
        ax3.set_ylabel('Frequency (%)')
        ax3.set_title('Operational Strategy Frequency')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Performance metrics
        metrics = {
            'Cost Reduction': improvement_pct,
            'Avg Gas Output': actions.get('avg_gas_output', 0),
            'Avg H2 Production': actions.get('avg_h2_production', 0)
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        ax4.barh(metric_names, metric_values, color=['red', 'orange', 'green'], alpha=0.7)
        ax4.set_xlabel('Value')
        ax4.set_title('Key Performance Metrics')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        performance_overview = self.run_dir / "visualizations" / "performance" / "performance_overview.png"
        plt.savefig(performance_overview, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved performance visualizations:")
        print(f"  - {performance_overview}")
    
    def create_data_analysis_visualizations(self, data):
        """Create data analysis and insights visualizations"""
        
        # 1. Market Data Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Energy Market Data Analysis', fontsize=16, fontweight='bold')
        
        # Price distribution
        if 'price_eur_mwh' in data.columns:
            ax1.hist(data['price_eur_mwh'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            ax1.axvline(data['price_eur_mwh'].mean(), color='red', linestyle='--', 
                       label=f'Mean: €{data["price_eur_mwh"].mean():.2f}/MWh')
            ax1.axvline(data['price_eur_mwh'].median(), color='green', linestyle='--',
                       label=f'Median: €{data["price_eur_mwh"].median():.2f}/MWh')
            ax1.set_xlabel('Price (€/MWh)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Electricity Price Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Renewable vs Load
        if 'total_renewable_mwh' in data.columns and 'load_mwh' in data.columns:
            ax2.scatter(data['load_mwh'], data['total_renewable_mwh'], alpha=0.7, color='green')
            ax2.plot([data['load_mwh'].min(), data['load_mwh'].max()], 
                    [data['load_mwh'].min(), data['load_mwh'].max()], 
                    'r--', label='Load = Renewable')
            ax2.set_xlabel('Load (MWh)')
            ax2.set_ylabel('Total Renewable (MWh)')
            ax2.set_title('Renewable Generation vs Load')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Hourly patterns
        if 'hour' in data.columns and 'price_eur_mwh' in data.columns:
            hourly_avg = data.groupby('hour')['price_eur_mwh'].mean()
            ax3.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='purple')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Average Price (€/MWh)')
            ax3.set_title('Average Price by Hour of Day')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(0, 24, 4))
        
        # Golden hours identification
        if 'is_golden_hour' in data.columns:
            golden_hours = data[data['is_golden_hour'] == True]['hour'].value_counts().sort_index()
            if not golden_hours.empty:
                ax4.bar(golden_hours.index, golden_hours.values, color='gold', alpha=0.8)
                ax4.set_xlabel('Hour of Day')
                ax4.set_ylabel('Count of Golden Hours')
                ax4.set_title('Golden Hours Distribution')
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, 'No Golden Hours Data Available', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Golden Hours Distribution')
        
        plt.tight_layout()
        data_analysis = self.run_dir / "visualizations" / "analysis" / "data_analysis.png"
        plt.savefig(data_analysis, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved data analysis visualizations:")
        print(f"  - {data_analysis}")
    
    def create_summary_report(self, metadata, training_data, performance_data):
        """Create a comprehensive summary report"""
        
        report = f"""
# RL Energy Cost Optimization - Run Summary Report

**Run Timestamp:** {metadata['run_info']['timestamp']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Cost Reduction Achieved:** {performance_data['cost_improvement_percent']:.1f}%
- **Total Cost Savings:** €{performance_data['total_cost_savings']:,.2f}
- **RL Agent Final Cost:** €{performance_data['rl_agent_cost']:,.2f}
- **Baseline Strategy Cost:** €{performance_data['baseline_cost']:,.2f}

## Training Performance

- **Episodes Completed:** {training_data['episodes_count']}
- **Initial Episode Cost:** €{training_data['initial_cost']:,.2f}
- **Final Episode Cost:** €{training_data['final_cost']:,.2f}
- **Best Episode Cost:** €{training_data['best_cost']:,.2f}
- **Training Improvement:** {training_data['cost_improvement']:.1f}%

## System Configuration

- **Gas Plant Capacity:** {metadata['system_config']['gas_capacity']:.1f} MWh
- **Storage Capacity:** {metadata['system_config']['storage_capacity']:.1f} MWh
- **H2 Production Capacity:** {metadata['system_config']['h2_production_capacity']:.1f} MWh
- **Gas Base Cost:** €{metadata['system_config']['gas_base_cost']:.2f}/MWh
- **H2 Selling Price:** €{metadata['system_config']['h2_selling_price']:.2f}/MWh

## Operational Strategy Analysis

- **Average Gas Output:** {performance_data['actions_summary'].get('avg_gas_output', 0):.1f} MWh
- **Gas Utilization Frequency:** {performance_data['actions_summary'].get('gas_utilization', 0)*100:.1f}%
- **Storage Charging Frequency:** {performance_data['actions_summary'].get('storage_charging_freq', 0)*100:.1f}%
- **H2 Production Frequency:** {performance_data['actions_summary'].get('h2_production_freq', 0)*100:.1f}%
- **Average H2 Production:** {performance_data['actions_summary'].get('avg_h2_production', 0):.1f} MWh

## Data Information

- **Data Source:** {metadata['data_info']['source_file']}
- **Data Shape:** {metadata['data_info']['data_shape'][0]} hours, {metadata['data_info']['data_shape'][1]} features
- **Golden Hours Identified:** {metadata['data_info']['golden_hours_count']}

## Key Insights

1. **Cost Efficiency:** The RL agent achieved {performance_data['cost_improvement_percent']:.1f}% cost reduction compared to baseline operations.

2. **Operational Optimization:** 
   - Gas plant operated {performance_data['actions_summary'].get('gas_utilization', 0)*100:.1f}% of the time
   - Storage system was actively used {performance_data['actions_summary'].get('storage_charging_freq', 0)*100:.1f}% of the time
   - Hydrogen production occurred {performance_data['actions_summary'].get('h2_production_freq', 0)*100:.1f}% of the time

3. **Learning Performance:** Training showed {training_data['cost_improvement']:.1f}% improvement from initial to final episodes.

## Files Generated

### Data Files
- `training_history.csv` - Episode-by-episode training data
- `training_results.json` - Training performance metrics
- `performance_results.json` - Final performance comparison

### Model Files
- `trained_agent.pkl` - Saved RL agent model
- `system_config.pkl` - System configuration

### Visualizations
- `training_overview.png` - Training performance charts
- `learning_analysis.png` - Detailed learning analysis
- `performance_overview.png` - Performance comparison charts
- `data_analysis.png` - Market data analysis

### Metadata
- `run_metadata.json` - Complete run configuration and info

---
*Report generated by RL Energy Optimization System*
"""

        report_file = self.run_dir / "reports" / "summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📋 Generated summary report: {report_file}")
        
        return report_file

def save_complete_analysis(agent, costs, rewards, test_results, config, data, source_file=""):
    """
    Complete function to save all analysis results and visualizations
    """
    
    print("\n💾 SAVING COMPLETE ANALYSIS RESULTS")
    print("="*50)
    
    # Initialize saver
    saver = AnalysisResultsSaver()
    
    # Prepare data info
    data_info = {
        "source_file": source_file,
        "shape": list(data.shape),
        "columns": list(data.columns),
        "golden_hours": data['is_golden_hour'].sum() if 'is_golden_hour' in data.columns else 0,
        "date_range": {
            "start": str(data.index.min()) if hasattr(data.index, 'min') else "N/A",
            "end": str(data.index.max()) if hasattr(data.index, 'max') else "N/A"
        }
    }
    
    # Training parameters
    training_params = {
        "episodes": len(costs),
        "state_size": 5,
        "action_size": 4,
        "learning_rate": 0.001,
        "epsilon_decay": 0.995
    }
    
    # Save all components
    print("📊 Saving metadata...")
    metadata = saver.save_training_metadata(config, training_params, data_info)
    
    print("📈 Saving training results...")
    training_data, performance_data = saver.save_training_results(agent, costs, rewards, test_results)
    
    print("🤖 Saving model artifacts...")
    saver.save_model_artifacts(agent, config)
    
    print("🎨 Creating training visualizations...")
    saver.create_training_visualizations(costs, rewards)
    
    print("📊 Creating performance visualizations...")
    saver.create_performance_visualizations(test_results, data)
    
    print("📈 Creating data analysis visualizations...")
    saver.create_data_analysis_visualizations(data)
    
    print("📋 Generating summary report...")
    report_file = saver.create_summary_report(metadata, training_data, performance_data)
    
    print(f"\n✅ ANALYSIS SAVE COMPLETE!")
    print(f"📁 Results saved to: {saver.run_dir}")
    print(f"📋 Summary report: {report_file}")
    
    return saver.run_dir, report_file