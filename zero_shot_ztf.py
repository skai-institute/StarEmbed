from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dask.distributed import Client
from lsdb import Catalog
import numpy.typing as npt
from dataclasses import dataclass
import os
from jaxtyping import Float, Bool
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule as L

# have to use lsdb==0.4.2, hats==0.4.3
from lsdb import read_hats
import lsdb
from astropy.coordinates import SkyCoord
from astropy import units as u

from sklearn.preprocessing import MinMaxScaler
import pickle



class TimeSeriesTestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        covariate_cols: List[str],
        future_covariate_cols: List[str],
        item_id_col: str = 'item_id',
        ctx: int = 200,
        horizon: int = 16
    ):
        """
        Initialize the TimeSeriesTestDataset

        :param df: the dataframe containing the time series data
        :type df: pd.DataFrame

        :param target_col: the name of the target column
        :type target_col: str

        :param covariate_cols: the columns of the covariates that only have history data
        :type covariate_cols: List[str]

        :param item_id_col: the name of the item_id column
        :type item_id_col: str

        :param ctx: the context length
        :type ctx: int

        :param horizon: the prediction length
        :type horizon: int
        """
        self.unique_ids = df[item_id_col].unique()
        self.grouped_data = {}
        
        # # Store each time series, use dataframe groupby to group by item_id
        # for item_id, group in df.groupby(item_id_col):
        #     self.grouped_data[item_id] = {
        #         'target_series': torch.FloatTensor(group[target_col].iloc[-ctx-horizon:-horizon].values),
        #         'covariates': torch.FloatTensor(group[covariate_cols].iloc[-ctx-horizon:-horizon].values),
        #         'future_covariates': torch.FloatTensor(group[future_covariate_cols].iloc[-ctx-horizon:].values),
        #         'timestamps': group['timestamp'].iloc[-ctx-horizon:].values # [batch_size, ctx+horizon, 1]
        #     }

        for item_id, group in df.groupby(item_id_col):
            self.grouped_data[item_id] = {
                'target_series': torch.FloatTensor(group[target_col].iloc[-ctx-horizon:-horizon].values),
                'covariates': torch.FloatTensor(group[covariate_cols].iloc[-ctx-horizon:-horizon].values),
                'timestamps': group['timestamp'].iloc[-ctx-horizon:].values # [batch_size, ctx+horizon, 1]
            }
    
    def __len__(self) -> int:
        return len(self.unique_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item_id = self.unique_ids[idx]
        return item_id, self.grouped_data[item_id]
    
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack individual time series into batched tensors
    """
    # Extract all target series and covariates
    target_series = [item[1]['target_series'] for item in batch]
    covariates = [item[1]['covariates'] for item in batch]
    # future_covariates = [item[1]['future_covariates'] for item in batch]
    timestamps = [item[1]['timestamps'] for item in batch]
    
    # Stack them into batched tensors
    return {
        'item_ids': [item[0] for item in batch],
        'target_series': torch.stack(target_series),  # [batch_size, sequence_length]
        'timestamps': timestamps,
        'covariates': torch.stack(covariates),         # [batch_size, sequence_length, n_covariates]
        # 'future_covariates': torch.stack(future_covariates)  # [batch_size, sequence_length, n_future_covariates]
    }

def infer_moirai(trn_dataloader: TorchDataLoader, 
                 covariate_cols: List[str], 
                 future_covariate_cols: List[str], 
                 CTX: int, 
                 PDT: int, 
                 PSZ: int, 
                 mode: str = 'moe'):
    """
    Make a zero-shot forecast using the moirai model.

    :param trn_dataloader: the dataloader for the inference data
    :type trn_dataloader: TorchDataLoader

    :param covariate_cols: the columns of the covariates that only have history data
    :type covariate_cols: List[str]

    :param future_covariate_cols: the columns of the covariates that have both history and future data
    :type future_covariate_cols: List[str]

    :param CTX: the context length
    :type CTX: int

    :param PDT: the prediction length
    :type PDT: int

    :param PSZ: the patch size
    :type PSZ: int

    :param mode: choose from MOIRAI-MOE or MOIRAI, choose from {'moe', 'regular'}
    :type mode: str
    """

    if mode == 'moe':
        model: L.LightningModule = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-base"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            past_feat_dynamic_real_dim=len(covariate_cols),
            feat_dynamic_real_dim=len(future_covariate_cols) if len(future_covariate_cols) > 0 else None,
            
        )
    else:
        SIZE = 'large'
        model: L.LightningModule = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            past_feat_dynamic_real_dim=len(covariate_cols),
            feat_dynamic_real_dim=len(future_covariate_cols) if len(future_covariate_cols) > 0 else None,
            
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    series_ids = []
    batched_predictions = []
    batched_predictions_90 = []
    batched_predictions_10 = []
    trn_timestamps = []
    with torch.no_grad():
        for batch in trn_dataloader:
            item_ids = batch['item_ids']
            series_ids.extend(item_ids)
            target_series: Float[torch.Tensor, "batch past_time tgt"] = batch['target_series'].unsqueeze(-1).to(device)
            past_observed_target: Bool[torch.Tensor, "batch past_time tgt"] = torch.ones_like(target_series, dtype=torch.bool)

            past_is_pad: Bool[torch.Tensor, "batch past_time"] = torch.zeros_like(target_series, dtype=torch.bool).squeeze(-1)

            past_feat_dynamic_real: Float[torch.Tensor, "batch past_time past_feat"] = batch['covariates'].to(device)
            past_observed_feat_dynamic_real: Float[torch.Tensor, "batch past_time past_feat"] = torch.ones_like(past_feat_dynamic_real, dtype=torch.float32)
            if past_feat_dynamic_real.shape[2] == 0:
                past_feat_dynamic_real = None
                past_observed_feat_dynamic_real = None


            feat_dynamic_real = None
            observed_feat_dynamic_real = None
            if len(future_covariate_cols) > 0:
                feat_dynamic_real: Float[torch.Tensor, "batch time feat"] = batch['future_covariates'].to(device)
                observed_feat_dynamic_real: Float[torch.Tensor, "batch time feat"] = torch.ones_like(feat_dynamic_real, dtype=torch.float32)

                if feat_dynamic_real.shape[2] == 0:
                    feat_dynamic_real = None
                    observed_feat_dynamic_real = None

            if PSZ == 'auto':
                # add horizon 0s to the past_target and past_observed_target in the second dimension
                target_series = torch.cat([target_series, torch.zeros(target_series.shape[0], PDT, 1).to(device)], dim=1)
                past_observed_target = torch.cat([past_observed_target, torch.zeros(past_observed_target.shape[0], PDT, 1, dtype=torch.bool).to(device)], dim=1)
                past_is_pad = torch.cat([past_is_pad, torch.zeros(past_is_pad.shape[0], PDT, dtype=torch.bool).to(device)], dim=1)

            preds = model.forward(
                past_target=target_series,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                past_feat_dynamic_real=past_feat_dynamic_real,
                past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
                feat_dynamic_real=feat_dynamic_real,
                observed_feat_dynamic_real=observed_feat_dynamic_real,
            )

            preds_median = np.median(preds.cpu().numpy(), axis=1)
            preds_mean = np.mean(preds.cpu().numpy(), axis=1)
            # also get the 90% quantile and 10% quantile
            preds_quantile_90 = np.quantile(preds.cpu().numpy(), 0.9, axis=1)
            preds_quantile_10 = np.quantile(preds.cpu().numpy(), 0.1, axis=1)
            batched_predictions.append(preds_mean)
            batched_predictions_90.append(preds_quantile_90)
            batched_predictions_10.append(preds_quantile_10)
            trn_timestamps.extend(batch['timestamps'])

    # merge batched_predictions and series_ids, and timestamps
    trn_preds_df = batch_predictions_to_dataframe(
        batched_predictions=batched_predictions,
        series_ids=series_ids,
        timestamps=trn_timestamps
    )

    preds_df_90 = batch_predictions_to_dataframe(
        batched_predictions=batched_predictions_90,
        series_ids=series_ids,
        timestamps=trn_timestamps
    )

    preds_df_10 = batch_predictions_to_dataframe(
        batched_predictions=batched_predictions_10,
        series_ids=series_ids,
        timestamps=trn_timestamps
    )

    print(trn_preds_df.head(2))
    
    print(f"=== inference done ===")

    return trn_preds_df, preds_df_90, preds_df_10

def batch_predictions_to_dataframe(
    batched_predictions: List[np.ndarray],
    series_ids: List[str],
    timestamps: List[pd.Timestamp],
    chunk_size: int = 1000  # Process this many series at a time
) -> pd.DataFrame:
    """
    Memory-efficient version that processes data in chunks.
    """
    all_rows = []
    n_timestamps = len(timestamps)
    batched_predictions = np.vstack(batched_predictions) # Shape: (total_series, horizon)
    horizon = batched_predictions.shape[1]

    print(f"len of timestamps: {len(timestamps)}")
    print(f"len of series_ids: {len(series_ids)}")
    print(f"batched_predictions.shape: {batched_predictions.shape}")
    
    # Process predictions in chunks
    for i in range(0, len(batched_predictions), chunk_size):
        # Get chunk of predictions
        chunk_preds = batched_predictions[i:i + chunk_size]
        chunk_series_ids = series_ids[i:i + chunk_size]
        chunk_timestamps = timestamps[i:i + chunk_size]

        # Flatten predictions
        flat_predictions = chunk_preds.flatten()  # Shape: (total_series * horizon,)
        
        # Flatten series IDs and repeat each ID horizon times
        flat_series_ids = np.repeat(chunk_series_ids, horizon) # Shape: (total_series * horizon,)

        # flat timestamps
        flat_timestamps = np.array(chunk_timestamps)[:, -horizon:].flatten() # Shape: (total_series * horizon,)

        # print all the shapes
        print(f"flat_series_ids.shape: {flat_series_ids.shape}")
        print(f"flat_timestamps.shape: {flat_timestamps.shape}")
        print(f"flat_predictions.shape: {flat_predictions.shape}")

        # Create the DataFrame
        df = pd.DataFrame({
            'series_id': flat_series_ids,
            'date': flat_timestamps,
            'prediction': flat_predictions
        })

        all_rows.append(df)
    
    # Create DataFrame from collected rows
    return pd.concat(all_rows)

@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    correlations: List[float]
    mse: List[float]
    r2: List[float]

    def print_summary(self) -> None:
        """Print summary statistics of all metrics."""
        print("\nAverage Metrics:")
        print(f"Correlation: {np.mean(self.correlations):.3f} ± {np.std(self.correlations):.3f}")
        print(f"MSE: {np.mean(self.mse):.3f} ± {np.std(self.mse):.3f}")
        print(f"R2: {np.mean(self.r2):.3f} ± {np.std(self.r2):.3f}")

def prepare_forecast_data(obj: pd.DataFrame, item_id: int, P: float) -> pd.DataFrame:
    """
    Prepare light curve data for forecasting.
    
    Args:
        obj: Raw light curve data
        item_id: Unique identifier for the star
        
    Returns:
        Preprocessed DataFrame ready for forecasting
    """

    # Convert MJD to datetime
    def mjd_to_datetime(mjd):
        # MJD epoch is November 17, 1858
        mjd_epoch = pd.Timestamp('1858-11-17')
        # Convert MJD to timedelta days and add to epoch
        return mjd_epoch + pd.TimedeltaIndex(mjd, unit='D')
    
    obj = (
        obj
        .query('band == "r"')
        [['mjd', 'mag', 'magerr']]
        .rename(columns={'mag': 'target', 'magerr': 'target_unc'})
        .assign(
            timestamp=lambda x: mjd_to_datetime(x['mjd']),
            item_id=item_id
        )
        .sort_values(by=['timestamp'])
        .assign(
            delta_t=lambda x: x['timestamp'].diff().dt.days
        )
        .fillna(0)
    )

    numeric_cols = ['target', 'target_unc']

    phase = (obj["mjd"].to_numpy() % P)
    obj['phase'] = phase
    obj = obj.drop(columns=['delta_t'])
    phase_col = 'phase'
    obj = obj.sort_values(by='phase')
    resample_points = 160

    # convert the series to phase folded
        # Create uniform phase grid for resampling
    phase_min = obj[phase_col].min()
    phase_max = obj[phase_col].max()
    uniform_phase = np.linspace(phase_min, phase_max, resample_points)
    
    # Initialize DataFrame for resampled data
    resampled_data = pd.DataFrame()
    resampled_data[phase_col] = uniform_phase
    
    # Resample each value column
    for col in numeric_cols:
        # Interpolate values onto uniform phase grid
        # Using cubic interpolation for smoother results
        resampled_values = np.interp(
            uniform_phase,
            obj[phase_col],
            obj[col],
            period=1.0  # Assuming phase is normalized to 1.0
        )
        resampled_data[col] = resampled_values

    obj = resampled_data
    

    # normalize target, target_unc and delta_t
    eps = 1e-8
    
    obj_mean = {}
    obj_std = {}

    for col in numeric_cols:
        obj_mean[col] = obj[col].mean()
        obj_std[col] = obj[col].std() + eps
        obj[col] = (obj[col] - obj_mean[col]) / obj_std[col]

    
    obj_mean = pd.Series(obj_mean)
    obj_std = pd.Series(obj_std)

    # concat 80 zeros to the end of obj, and 80 ones to the end of obj_mean and obj_std
    obj = pd.concat([obj, pd.DataFrame({'target': np.zeros(80), 'target_unc': np.zeros(80), 'phase': np.zeros(80)})], ignore_index=True)
    obj_mean = pd.concat([obj_mean, pd.Series(np.zeros(80))], ignore_index=True)
    obj_std = pd.concat([obj_std, pd.Series(np.ones(80))], ignore_index=True)

    obj_mean['item_id'] = item_id
    obj_std['item_id'] = item_id

    obj['item_id'] = item_id

    # create a fake timestamp column starting from 2022-01-01
    obj['timestamp'] = pd.date_range(start='2022-01-01', periods=len(obj), freq='D')
    
    return obj, obj_mean, obj_std

def calculate_star_metrics(
    history: pd.DataFrame, 
    predictions: pd.DataFrame
) -> Tuple[float, float, float]:
    """
    Calculate performance metrics for a single star's predictions.
    
    Args:
        history: Historical light curve data
        predictions: Predicted light curve data
        
    Returns:
        Tuple of (correlation, mse, r2_score)
    """
    merged_df = pd.merge(
        history,
        predictions,
        left_on='timestamp',
        right_on='date',
        how='inner'
    )
    
    if len(merged_df) == 0:
        return 0.0, 0.0, 0.0
        
    correlation, _ = pearsonr(merged_df['target'], merged_df['prediction'])
    mse = mean_squared_error(merged_df['target'], merged_df['prediction'])
    r2 = r2_score(merged_df['target'], merged_df['prediction'])
    
    return correlation, mse, r2

def plot_star_forecast(
    ax: plt.Axes,
    context_length: int,
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    predictions_90: pd.DataFrame,
    predictions_10: pd.DataFrame,
    metrics: Tuple[float, float, float],
    item_id: int,
    is_first: bool = False
) -> None:
    """
    Plot a single star's forecast comparison.
    
    Args:
        ax: Matplotlib axes to plot on
        history: Historical light curve data
        predictions: Predicted light curve data
        metrics: Tuple of (correlation, mse, r2)
        item_id: Star identifier
        is_first: Whether this is the first plot (for legend)
    """
    correlation, mse, _ = metrics
    ax.plot(history['timestamp'].iloc[-len(predictions)-context_length:], history['target'].iloc[-len(predictions)-context_length:], 
            label='History', alpha=0.6)
    ax.plot(predictions['date'], predictions['prediction'], 
            label='Predicted', linestyle='--')
    ax.set_title(f'Star {item_id}\nCorr: {correlation:.2f}, MSE: {mse:.2f}')
    ax.tick_params(axis='x', rotation=45)
    if is_first:
        ax.legend()
    
    # plot the 90% quantile
    ax.plot(predictions_90['date'], predictions_90['prediction'], 
            label='90% Quantile', linestyle='--')
    # plot the 10% quantile
    ax.plot(predictions_10['date'], predictions_10['prediction'], 
            label='10% Quantile', linestyle='--')

def analyze_multiple_stars_batched(
    raw_catalog: Catalog,
    varstars: pd.DataFrame,
    n_stars: int = 20,
    seed: int = 42,
    use_cache: bool = True
) -> ForecastMetrics:
    """
    Analyze multiple stars using batched processing for efficient forecasting.
    
    Args:
        raw_catalog: ZTF catalog
        varstars: DataFrame containing variable stars information
        n_stars: Number of stars to analyze
        seed: Random seed for reproducibility
        
    Returns:
        ForecastMetrics containing correlation, MSE, and R2 scores
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random_indices = np.random.randint(0, len(varstars), n_stars)
    print(f"random_indices: {random_indices}")
    # random_indices = [15795, 860, 38158, 44732, 11284, 6265, 16850, 37194, 21962, 44131, 16023, 41090, 1685, 769, 2433, 5311, 37819, 39188, 17568, 19769]
    # random_indices = [9710]
    selected_stars = varstars.iloc[random_indices]
    print(f"selected_stars: {selected_stars[['ID', 'RAdeg', 'Decdeg']]}")
    # Fetch all light curves in parallel
    # load the cache if exists
    cache_path = f"all_objects_47054_None.pkl"
    with open(cache_path, 'rb') as f:
        all_objects = pickle.load(f)
    
    # Process all light curves
    all_forecasts = []
    all_forecasts_mean = []
    all_forecasts_std = []
    # Model configuration
    future_covariate_cols = []
    covariate_cols = ['target_unc']
    CTX = 160  # Context window
    PDT = 80  # Prediction horizon
    PSZ = 16   # Patch size
    for idx, obj in enumerate(all_objects):
        if len(obj) < (CTX + PDT):  # Minimum required for context + prediction
            continue
        print(f"len of star {idx + 1}: {len(obj)}")
        P = selected_stars.iloc[idx]['P']
        df_forecast, df_forecast_mean, df_forecast_std = prepare_forecast_data(obj, idx + 1, P)
        if len(df_forecast) < (CTX + PDT):
            continue
        all_forecasts.append(df_forecast)
        all_forecasts_mean.append(df_forecast_mean)
        all_forecasts_std.append(df_forecast_std)
    
    # Combine all forecasts for batch processing
    combined_df = pd.concat(all_forecasts, ignore_index=True)
    combined_df_mean = pd.DataFrame([s.to_dict() for s in all_forecasts_mean])
    combined_df_std = pd.DataFrame([s.to_dict() for s in all_forecasts_std])
    

    
    # Prepare dataset and dataloader
    tst_dataset = TimeSeriesTestDataset(
        df=combined_df,
        target_col='target',
        covariate_cols=covariate_cols,
        future_covariate_cols=future_covariate_cols,
        item_id_col='item_id',
        ctx=CTX,
        horizon=PDT
    )
    
    dataloader = TorchDataLoader(
        tst_dataset,
        batch_size=len(all_forecasts),  # Process all stars in single batch
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False
    )
    
    # Generate predictions
    preds_df, preds_df_90, preds_df_10 = infer_moirai(
        trn_dataloader=dataloader,
        covariate_cols=covariate_cols,
        future_covariate_cols=future_covariate_cols,
        CTX=CTX,
        PDT=PDT,
        PSZ=PSZ,
        mode='regular'
    )
    
    # Visualization setup
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    fig_phase, axes_phase = plt.subplots(4, 5, figsize=(20, 16))  # 新增相位图
    axes_phase = axes_phase.flatten()
    
    # Calculate metrics and create plots for each star
    metrics = ForecastMetrics([], [], [])
    numeric_cols = ['target', 'target_unc']
    for idx, item_id in enumerate(preds_df['series_id'].unique()):
        star_history = combined_df[combined_df['item_id'] == item_id]
        star_history_mean = combined_df_mean[combined_df_mean['item_id'] == item_id]
        star_history_std = combined_df_std[combined_df_std['item_id'] == item_id]
        for col in numeric_cols:
            star_history[col] = star_history[col] * star_history_std[col].values[0] + star_history_mean[col].values[0]
        star_preds = preds_df[preds_df['series_id'] == item_id]
        star_preds['prediction'] = star_preds['prediction'] * star_history_std['target'].values[0] + star_history_mean['target'].values[0]
        star_preds_90 = preds_df_90[preds_df_90['series_id'] == item_id]
        star_preds_90['prediction'] = star_preds_90['prediction'] * star_history_std['target'].values[0] + star_history_mean['target'].values[0]
        star_preds_10 = preds_df_10[preds_df_10['series_id'] == item_id]
        star_preds_10['prediction'] = star_preds_10['prediction'] * star_history_std['target'].values[0] + star_history_mean['target'].values[0]
        P = selected_stars.iloc[idx]['P']

        # convert from flux back to magnitude
        # Flux in microJy
        # ndf['flux'] = 3.631e9 * 10**(ndf['mag'] / -2.5)
        # # Flux uncertainty in microJy
        # ndf['fluxerr'] = ndf['flux'] * np.log(10) * ndf['magerr'] / 2.5
        star_history['target_unc'] = star_history['target_unc'] * 2.5 / star_history['target'] / np.log(10)
        star_history['target'] = np.log10(star_history['target'] / 3.631e9) * (-2.5)
        star_preds['prediction'] = np.log10(star_preds['prediction'] / 3.631e9) * (-2.5)
        star_preds_90['prediction'] = np.log10(star_preds_90['prediction'] / 3.631e9) * (-2.5)
        star_preds_10['prediction'] = np.log10(star_preds_10['prediction'] / 3.631e9) * (-2.5)


        # Calculate metrics
        correlation, mse, r2 = calculate_star_metrics(star_history, star_preds)
        metrics.correlations.append(correlation)
        metrics.mse.append(mse)
        metrics.r2.append(r2)
        
        # Create plot
        plot_star_forecast(
            axes[idx], 
            CTX,
            star_history, 
            star_preds, 
            star_preds_90,
            star_preds_10,
            (correlation, mse, r2),
            item_id,
            idx == 0
        )

        # # also plot the ground truth and predictions' period folded plots
        # # We can replot the light curve but "fold" the light curve by the star's period
        # # Now we can see that this light curve is clearly dynamic
        # # Should be a much more interesting case for forecasting
        # # I had some trouble bulk querying for cross matches between the varstar list and ZTF DR14 but there are some examples for doing so online
        # print("Star's period", P, "days")

        # ax_phase = axes_phase[idx]
        # P_value = P
        
        # phase = (star_history["mjd"].to_numpy() % P_value)
        # phase_pred = (star_history["mjd"].iloc[-len(star_preds):].to_numpy() % P_value)

        # ax_phase.errorbar(phase, star_history["target"], 
        #                  yerr=star_history["target_unc"], 
        #                  label='ground truth', 
        #                  color='g', fmt='x')
        
        # ax_phase.errorbar(phase_pred, star_preds['prediction'], 
        #                  label='predictions',
        #                  color='b', fmt='o')
        
        # # plot the 90% quantile
        # ax_phase.errorbar(phase_pred, star_preds_90['prediction'], 
        #                  label='90% Quantile',
        #                  color='r', fmt='o')
        # # plot the 10% quantile
        # ax_phase.errorbar(phase_pred, star_preds_10['prediction'], 
        #                  label='10% Quantile',
        #                  color='y', fmt='o')

        # ax_phase.set_title(f"Item {item_id} (P={P_value:.2f}d)")
        # ax_phase.legend()
        # ax_phase.set_xlabel("Phase")
        # ax_phase.set_ylabel("Magnitude")
        # ax_phase.invert_yaxis()
    

    fig.tight_layout()
    fig.savefig(f'multiple_stars_forecast_batched_{n_stars}_{seed}_phase_folded.png')
    plt.close(fig)
    

    # fig_phase.tight_layout()
    # fig_phase.savefig(f'multiple_stars_phase_folded_{n_stars}_{seed}_phase_folded.png')
    # plt.close(fig_phase)
    
    # Print metric summary
    metrics.print_summary()
    
    return metrics

if __name__ == "__main__":
    ZTF_SOURCES = "https://data.lsdb.io/hats/ztf_dr14/ztf_source"
    
    # Load ZTF catalog
    raw_catalog = read_hats(ZTF_SOURCES)
    
    # Load and process variable stars data
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 
                   'magV', 'P', "Amp", "class", 'flag']
    """
            ID  RAh  RAm    RAs  Decsign  DEm   DEs   magV         P   Amp class flag
    0  CSS_J000031.5-084652    0    0  31.50       -8   46  52.3  14.14  0.404185  0.12     1  NaN
    1  CSS_J000036.9+412805    0    0  36.94       41   28   5.7  17.39  0.274627  0.73     1  NaN
    """
    varstars = pd.read_csv(
        "CSDR1_varstars.txt",
        header=33,
        sep='\s+',
        names=column_names
    )
    
    # Convert coordinates
    RAhms = [f"{row.RAh}h{row.RAm}m{row.RAs}s" for _, row in varstars.iterrows()]
    Decdms = [f"{row.Decsign}d{row.DEm}m{row.DEs}s" for _, row in varstars.iterrows()]
    coords = SkyCoord(ra=RAhms, dec=Decdms, frame='icrs')
    varstars['RAdeg'] = coords.ra.deg
    varstars['Decdeg'] = coords.dec.deg
    
    # Run analysis
    metrics = analyze_multiple_stars_batched(raw_catalog, varstars, n_stars=20, use_cache=True, seed=50)