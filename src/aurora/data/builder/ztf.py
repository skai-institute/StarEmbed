#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from pathlib import Path
from typing import Any, Generator, Dict

import datasets
import pyarrow.parquet as pq
from datasets import Features, Sequence, Value
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable, Iterable

import datasets
import pandas as pd


from uni2ts.common.env import env
from ._base import DatasetBuilder
from uni2ts.data.dataset import TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation
from torch.utils.data import Dataset
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import argparse


def read_merge_data(data_dir: str,
                    offset: int = 80) -> pd.DataFrame:
    """
    Read all the csv files in the data_dir and merge them into a single pandas DataFrame.

    :param data_dir: The directory containing the csv files.
    :type data_dir: str

    :return: A pandas DataFrame containing the merged data.
    :rtype: pd.DataFrame
    """
    data_files = list(Path(data_dir).glob('*.csv'))[:offset]
    df = pd.concat([pd.read_csv(file) for file in data_files], ignore_index=True)
    return df



class FrequencyModifier(Transformation):
    """Transformation that overwrites the 'freq' field with a specified value."""
    
    def __init__(self, new_freq: str = 'D'):
        self.new_freq = new_freq
        
    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy of the item to avoid modifying the original
        # modified_item = dict(item)
        # Overwrite the 'freq' field
        item['freq'] = self.new_freq
        return item

class OnlineZScoreNormalization(Transformation):
    def __init__(self, fields: list[str] = ["target", "past_feat_dynamic_real", "feat_dynamic_real"]):
        self.fields = fields

    def __call__(self, data: dict) -> dict:
        eps = 1e-8
        # 对每个指定字段进行归一化
        # print(f"data keys: {data.keys()}")
        for field in self.fields:
            arr = data[field].copy()  # 避免修改原始数据
            mean = np.mean(arr)
            std = np.std(arr) + eps
            min_value = np.min(arr)
            max_value = np.max(arr)
            data[field] = (arr - mean) / (std)
            # 保存统计量到metadata（不修改原始数据集结构）
            data[f"{field}_scale"] = np.array([mean, std, min_value, max_value], dtype=np.float32)
        return data

@dataclass
class ZTFDatasetBuilder(DatasetBuilder):

    dataset: str
    storage_path: str = "/scratch/wlk5936/ztf"

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, 
                      dataset: str,
                      dataset_filename: str  = 'dr14_clean_100_partitions2'):
        def _from_long_dataframe(
            df: pd.DataFrame,
            offset: Optional[int] = None,
            date_offset: Optional[pd.Timestamp] = None,
            freq: str = "D", # decided by ad-hoc by looking at the data
        ) -> tuple[Callable[[], Iterable[dict[str, Any]]], Features]:
            items = df.item_id.unique()
            print(f"Total items: {len(items)}")
            def example_gen_func() -> Generator[dict[str, Any], None, None]:
                # get a progress bar
                for item_id in tqdm(items):
                    print(f"Processing item_id: {item_id}")
                    item_df = df.query(f'item_id == "{item_id}"').drop("item_id", axis=1)
                    if offset is not None:
                        item_df = item_df.iloc[:offset]
                    elif date_offset is not None:
                        item_df = item_df[item_df.index <= date_offset]
                    yield {
                        "target": item_df['target'].to_numpy(),
                        "past_feat_dynamic_real": item_df['target_unc'].to_numpy(),
                        "feat_dynamic_real": item_df['delta_t'].to_numpy(),
                        "start": item_df['timestamp'].iloc[0],
                        "freq": freq,
                        "item_id": item_id,
                    }

            features = Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                    past_feat_dynamic_real=Sequence(Value("float32")),
                    feat_dynamic_real=Sequence(Value("float32")),
                )
            )

            return example_gen_func, features

        df = read_merge_data("dr14_clean_100_partitions2", offset=80)
        example_gen_func, features = _from_long_dataframe(df)
        hf_dataset = datasets.Dataset.from_generator(example_gen_func, features=features, cache_dir=env.HF_CACHE_PATH)
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(dataset_path=env.LOTSA_V1_PATH / dataset, num_proc=os.cpu_count()-2)



    def load_dataset(
        self, transform_map: dict[str | type, Callable[..., Transformation]]
    ) -> Dataset:
        base_transform = transform_map[self.dataset]()

        # Add frequency modifier before normalization
        freq_modifier = FrequencyModifier(new_freq='D')
        # Chain transformations: freq_modifier -> normalization -> base_transform
        transform = freq_modifier + OnlineZScoreNormalization() + base_transform
        # # 添加在线归一化层（在已有transform之前）
        # transform = OnlineZScoreNormalization() + base_transform
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform,
            dataset_weight=1.0
        )
    
# test the databuilder and dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["wide", "long", "wide_multivariate"],
        default="wide",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--date_offset",
        type=str,
        default=None,
    )
    # Define the `freq` argument with a default value. Use this value as 'freq' if 'freq' is None.
    parser.add_argument(
        "--freq",
        default="H",  # Set the default value
        help="The user specified frequency",
    )

    args = parser.parse_args()
    print(f"path: {args.file_path}")
    SimpleDatasetBuilder(dataset=args.dataset_name).build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        offset=args.offset,
        date_offset=pd.Timestamp(args.date_offset) if args.date_offset else None,
        freq=args.freq,
    )

    if args.offset is not None or args.date_offset is not None:
        SimpleEvalDatasetBuilder(
            f"{args.dataset_name}_eval",
            offset=None,
            windows=None,
            distance=None,
            prediction_length=None,
            context_length=None,
            patch_size=None,
        ).build_dataset(
            file=Path(args.file_path), dataset_type=args.dataset_type, freq=args.freq
        )

    builder = ZTFDatasetBuilder(dataset="dr14_clean_100_partitions2_offset80_5")
    builder.build_dataset("ztf")

