import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import numpy as np


def predict(df, target, feature_cols, CTX, PDT, TEST, SIZE="large", BSZ=32, PSZ="auto", plot=False):

    ds = PandasDataset(df, target=target, past_feat_dynamic_real=feature_cols)

    # if len(targets) == 1:
    #     # Convert into GluonTS dataset
    #     ds = PandasDataset(df, target=targets[0], past_feat_dynamic_real=feature_cols)
    # else:
    #     ds = PandasDataset(df, target=targets, past_feat_dynamic_real=feature_cols)
    #     # Group time series into multivariate dataset
    #     grouper = MultivariateGrouper(len(ds))
    #     ds = grouper(ds)
        

    # Split into train/test set
    train, test_template = split(
        ds, offset=-TEST
    )  # assign last TEST time steps as test set

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=TEST // PDT,  # number of windows in rolling window evaluation
        distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
    )

    # Prepare pre-trained model by downloading model weights from huggingface hub
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(test_data.input)
    print(test_data.input)

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)

    print(input_it)
    inp = next(input_it)
    label = next(label_it)
    forecast = next(forecast_it)

    # try:
    #     while True:
    #         print('----')
    #         label = next(label_it)
    #         print(label['item_id'])
    #         print(len(label['target']))
    #         print(len(next(input_it)['target']))
    #         print(next(forecast_it).start_date)
    # except StopIteration:
    #     pass

    # print(len(list(label_it)))

    # print(forecast.shape)

    if plot:
        plot_single(
            inp, 
            label, 
            forecast, 
            context_length=200,
            name="pred",
            show_label=False,
        )

        plt.tight_layout()  # adjusts spacing between subplots
        plt.savefig('forecasts2.png')

    return forecast, label, inp


def predict_without_split(df, target, feature_cols, CTX, PDT, TEST, SIZE="large", BSZ=32, PSZ="auto", plot=False):

    if not isinstance(df, PandasDataset):
        ds = PandasDataset(df, target=target, past_feat_dynamic_real=feature_cols)
    else:
        ds = df

    # Prepare pre-trained model by downloading model weights from huggingface hub
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(ds)

    input_it = iter(ds)
    forecast_it = iter(forecasts)

    print(input_it)
    inp = next(input_it)
    forecast = next(forecast_it)

    return forecast, None, inp


if __name__ == "__main__":

    SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
    PDT = 20  # prediction length: any positive integer
    CTX = 200  # context length: any positive integer
    PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 32  # batch size: any positive integer
    TEST = 20  # test set length: any positive integer

    # Read data into pandas DataFrame
    url = (
        "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
        "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
    )
    p = "/root/uni2ts-main/BE.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df['location'] = 'Belgian'
    print(df.columns)

    target = ' Prices'
    # target = [' Prices', ' Generation forecast']
    feature_cols = [' Generation forecast', ' System load forecast', ' Location']
    feature_cols = []

    df.to_csv('BE_with_static.csv')



    predict(df, target, feature_cols, CTX, PDT, TEST, plot=True)