# 2 x 2
# class 1
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/1/120620174173493992_g.png
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/1/129030693624086743_g.png
# class 5
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/5/97211305511826371_g.png # 174
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/5/130603207444200557_g.png # 397
# /home/magics/hdd/sky_ws/ebsim_ws/data/lsdb/5/135293431680766623_g.png

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from astropy.time import Time
from chronos.base import BaseChronosPipeline
from sklearn.cluster import KMeans


def seed_everything(seed=42):
    """
    Seed everything for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_lightcurve(processed_df, model_name="amazon/chronos-t5-small", context_percent=0.8, split_point=None, fig_dir='./data/debug/eval'):
    """
    Predict magnitude values for each star in the processed DataFrame
    
    Args:
        processed_df: DataFrame with two-level index (ps1_objid, phase)
        model_name: Name of the Chronos model to use
        context_percent: Percentage of data to use as context (default: 0.8)
        split_point: Point to split the data into context and target, it will override context_percent (default: None)

    Returns:
        dict: Dictionary containing predictions for each star
            {star_id: {'context': array, 
                      'target': array,
                      'mean': array, 
                      'quantiles': array}}
    """
    os.makedirs(fig_dir, exist_ok=True)  # equivalent to mkdir -p

    # Initialize Chronos pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    
    predictions = {}
    
    # Process each star
    for star_id in processed_df.index.get_level_values('ps1_objid').unique():
        # Get magnitude data for this star
        star_data = processed_df.loc[star_id]
        
        # Get magnitudes
        mags = star_data['mag'].values
        
        # Calculate split point
        n_points = len(mags)
        if split_point is None:
            split_point = int(context_percent * n_points)
        else:
            split_point = int(split_point)
        
        # Split into context and target
        context = mags[:split_point]
        target = mags[split_point:]
        
        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        # Predict
        quantiles, mean = pipeline.predict_quantiles(
            context=context_tensor,
            prediction_length=len(target),
            quantile_levels=[0.1, 0.5, 0.9],
        )

        # Squeeze predictions to remove extra dimensions
        mean = mean.squeeze().cpu().numpy()
        quantiles = quantiles.squeeze().cpu().numpy()
        
        # Store results
        predictions[star_id] = {
            'context': context,
            'target': target,
            'mean': mean,
            'quantiles': quantiles
        }
        
        # Calculate MSE dropping NaN values
        valid_mask = ~np.isnan(target) & ~np.isnan(mean)
        target_clean = target[valid_mask]
        mean_pred_clean = mean[valid_mask]
        mse = np.mean((target_clean - mean_pred_clean)**2)

        # Plot prediction for this star
        plt.figure(figsize=(12, 6))
        phases = star_data.index.get_level_values('phase')
        context_phases = phases[:split_point]
        target_phases = phases[split_point:]
        
        # Plot context
        plt.plot(context_phases, context, 'b.', label='Context', alpha=0.5)
        # Plot target
        plt.plot(target_phases, target, 'g.', label='Target', alpha=0.5)
        # Plot prediction
        plt.plot(target_phases, mean, 'r-', label='Prediction', linewidth=2)
        # Plot quantiles
        plt.fill_between(target_phases, 
                        quantiles[:, 0], 
                        quantiles[:, 2], 
                        color='r', alpha=0.2, label='90% Confidence')

        
        plt.title(f'Star {star_id} (Class {star_data["class"].iloc[0]}) Prediction - MSE: {mse:.4f}')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.legend()
        # plt.gca().invert_yaxis()  # Invert y-axis for magnitude
        plt.savefig(f'{fig_dir}/prediction_{star_id}.png')
        plt.close()
    
    return predictions


def main(args):
    """
    Main function to run forecast evaluation.
    
    Args:
        args: Command line arguments containing:
            model_path: Path to Chronos model to use
            data_path: Path to processed light curve CSV file
            output_dir: Directory to save prediction results and plots
    """
    # Initialize random seed
    seed_everything()
    
    # Load processed data
    print(f"Loading data from {args.data_path}...")
    processed_df = pd.read_csv(args.data_path)
    processed_df = processed_df.set_index(['ps1_objid', 'phase']).sort_index()
    star_ids_list = processed_df.index.get_level_values('ps1_objid').unique()
    print(f"Loaded data for {len(star_ids_list)} stars")
    
    # Make predictions
    print(f"Making predictions using model: {args.model_path}")
    predictions = predict_lightcurve(
        processed_df, 
        model_name=args.model_path,  # Using model_path instead of model_name
        context_percent=0.8,    # This won't be used because split_point is provided
        split_point=-32,        # Using fixed split point of -32
        fig_dir=f"{args.output_dir}/plots"
    )
    
    # Print some information about the processed data
    print("\nData summary:")
    for i, star_id in enumerate(star_ids_list):
        if i >= 5:  # Limit output to first 5 stars
            break
        star_data = processed_df.loc[star_id]
        print(f"Star {star_id}:")
        print(f"  Phase range: {star_data.index.min():.3f} to {star_data.index.max():.3f}")
        print(f"  Number of points: {len(star_data)}")
    
    # Print prediction metrics
    print("\nPrediction metrics:")
    mse_values = []
    for star_id, pred in predictions.items():
        target = pred['target']
        mean_pred = pred['mean']
        # Drop NaN values from both arrays
        valid_mask = ~np.isnan(target) & ~np.isnan(mean_pred)
        target_clean = target[valid_mask]
        mean_pred_clean = mean_pred[valid_mask]
        # Calculate MSE only on valid values
        if len(target_clean) > 0:
            mse = np.mean((target_clean - mean_pred_clean)**2)
            mse_values.append(mse)
            print(f"Star {star_id} MSE: {mse:.4f}")
    
    # Print average MSE if we have any valid predictions
    if mse_values:
        avg_mse = np.mean(mse_values)
        print(f"\nAverage MSE across all stars: {avg_mse:.4f}")
    
    print(f"\nPrediction plots saved to: {args.output_dir}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Forecast light curves using Chronos model")
    parser.add_argument("--model-path", type=str, default="amazon/chronos-t5-small",
                        help="Path to Chronos model to use")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to processed light curve CSV file")
    parser.add_argument("--output-dir", type=str, default="./data/eval/forecast",
                        help="Directory to save prediction results and plots")
    
    args = parser.parse_args()
    
    # Check if data path exists before proceeding
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Call main with args
    main(args)