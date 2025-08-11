"""
Main module for Financial Security AI Model dashboard
"""

import os
import argparse
from dashboard_integration import FinancialSecurityDashboard

def main():
    parser = argparse.ArgumentParser(description='Financial Security AI Model Dashboard')
    parser.add_argument('--model', type=str, default="upstage/SOLAR-10.7B-Instruct-v1.0",
                        help='Model to use for inference (default: SOLAR-10.7B-Instruct)')
    parser.add_argument('--test_file', type=str, default="/workspace/uploads/test.csv",
                        help='Path to test CSV file (default: /workspace/uploads/test.csv)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to results CSV file (optional)')
    parser.add_argument('--run_inference', action='store_true',
                        help='Run inference on sample data')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples for inference (default: 5)')
    
    args = parser.parse_args()
    
    print(f"Initializing Financial Security AI Model Dashboard")
    print(f"Model: {args.model}")
    print(f"Test file: {args.test_file}")
    
    # Initialize dashboard
    dashboard = FinancialSecurityDashboard(model_path=args.model)
    
    # Load and analyze data
    print("Loading and analyzing data...")
    dashboard.load_data(test_path=args.test_file, results_path=args.results_file)
    
    # Generate visualizations
    print("Generating visualizations...")
    dashboard.generate_visualizations()
    
    # Update dashboard components
    print("Updating dashboard components...")
    dashboard.update_dashboard_components()
    
    # Run inference if requested
    if args.run_inference:
        print(f"Running inference on {args.num_samples} samples...")
        dashboard.load_inference_model()
        results = dashboard.run_inference_sample(num_samples=args.num_samples)
        print(f"Inference completed on {len(results)} samples")
    
    print("\nDashboard updated successfully!")
    print("Access the dashboard through the development server.")

if __name__ == "__main__":
    main()