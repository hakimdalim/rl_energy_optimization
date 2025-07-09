# complete_rl_analysis.py - All-in-one script for RL optimization with auto-save
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

def run_complete_rl_analysis():
    """
    Complete RL analysis with automatic result saving
    """
    
    print("🚀 COMPLETE RL ENERGY OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Configuration
    pickle_path = "./preprocessed_smard_data/smard_processed_20250709_085507.pkl"
    episodes = 50
    
    # Step 1: Load and check data
    print(f"📂 Step 1: Loading data from {pickle_path}")
    
    try:
        # Import the updated run script
        from run_smard_optimization import run_smard_rl_optimization
        from save_analysis_results import save_complete_analysis
        
        # Run optimization
        agent, costs, rewards, test_results = run_smard_rl_optimization(
            pickle_path, episodes=episodes, save_results=False  # We'll save manually
        )
        
        # Load data for saving
        from run_smard_optimization import load_and_adapt_smard_data, create_optimized_config_for_smard
        df = load_and_adapt_smard_data(pickle_path)
        config = create_optimized_config_for_smard(df)
        
        # Save complete analysis
        print(f"\n💾 Step 2: Saving complete analysis...")
        results_dir, report_file = save_complete_analysis(
            agent, costs, rewards, test_results, config, df, pickle_path
        )
        
        # Display summary
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"📊 Performance Summary:")
        print(f"  Cost Reduction: {test_results['improvement_percent']:.1f}%")
        print(f"  Total Savings: €{test_results['cost_savings']:,.2f}")
        print(f"  RL Cost: €{test_results['rl_cost']:,.2f}")
        print(f"  Baseline Cost: €{test_results['baseline_cost']:,.2f}")
        
        print(f"\n📁 Results Saved To:")
        print(f"  Directory: {results_dir}")
        print(f"  Report: {report_file}")
        
        print(f"\n📊 Generated Files:")
        print(f"  🎨 Visualizations in: {results_dir}/visualizations/")
        print(f"  📈 Training plots: training_overview.png, learning_analysis.png")
        print(f"  📊 Performance plots: performance_overview.png")
        print(f"  📋 Data analysis: data_analysis.png")
        print(f"  📄 Summary report: summary_report.md")
        print(f"  🤖 Trained model: models/trained_agent.pkl")
        print(f"  📊 Data files: data/*.csv, data/*.json")
        
        # Open results folder (if on supported system)
        try:
            import os
            import platform
            if platform.system() == "Linux":
                os.system(f"xdg-open {results_dir}")
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open {results_dir}")
            elif platform.system() == "Windows":
                os.system(f"explorer {results_dir}")
            print(f"📁 Opening results folder...")
        except:
            pass
        
        return results_dir, test_results
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {pickle_path}")
        print(f"🔧 Please check the file path or run with sample data")
        
        # Run with sample data
        print(f"\n🚀 Running with sample data for demonstration...")
        return run_sample_analysis()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_sample_analysis():
    """Run analysis with sample data if real data is not available"""
    
    print(f"\n🧪 RUNNING SAMPLE DATA ANALYSIS")
    print("-"*40)
    
    try:
        from energy_cost_rl import test_with_smard_data
        test_with_smard_data()
        
        print(f"\n✅ Sample analysis completed!")
        print(f"💡 To run with your real data:")
        print(f"   1. Update pickle_path in the script")
        print(f"   2. Ensure all files are in the same directory")
        print(f"   3. Run: python complete_rl_analysis.py")
        
    except Exception as e:
        print(f"❌ Sample analysis failed: {e}")
    
    return None, None

def show_analysis_guide():
    """Show guide for interpreting the analysis results"""
    
    guide = """
📊 ANALYSIS RESULTS GUIDE
========================

📁 Folder Structure:
├── metadata/
│   └── run_metadata.json          # Complete run configuration
├── data/
│   ├── training_history.csv       # Episode-by-episode data
│   ├── training_results.json      # Training metrics
│   └── performance_results.json   # Final performance
├── models/
│   ├── trained_agent.pkl          # Saved RL agent
│   └── system_config.pkl          # System configuration
├── visualizations/
│   ├── training/
│   │   ├── training_overview.png  # Training progress
│   │   └── learning_analysis.png  # Detailed learning
│   ├── performance/
│   │   └── performance_overview.png # Performance comparison
│   └── analysis/
│       └── data_analysis.png      # Market data analysis
└── reports/
    └── summary_report.md           # Complete summary

🎯 Key Metrics to Look For:
- Cost Reduction %: Higher is better (target: >10%)
- Total Savings: Absolute cost savings in €
- Gas Utilization: How often gas plant operates
- Storage Activity: Battery charging/discharging frequency
- H2 Production: Hydrogen production frequency

📊 Important Charts:
1. Training Overview: Shows learning progress
2. Performance Comparison: RL vs Baseline costs
3. Action Analysis: Operational strategy patterns
4. Market Analysis: Price and renewable patterns

💡 Optimization Tips:
- More episodes = better learning (try 100-200)
- Adjust system capacities based on your actual plant
- Monitor golden hours utilization
- Check storage charging during low prices
    """
    
    print(guide)

if __name__ == "__main__":
    # Run complete analysis
    results_dir, test_results = run_complete_rl_analysis()
    
    if results_dir:
        print(f"\n📚 Analysis complete! Check your results in:")
        print(f"   {results_dir}")
        
        # Show guide
        print(f"\n" + "="*60)
        show_analysis_guide()
    
    else:
        print(f"\n⚠️  Analysis could not be completed.")
        print(f"💡 Check the error messages above for troubleshooting.")
    
    print(f"\n🎉 Thank you for using the RL Energy Optimization System!")