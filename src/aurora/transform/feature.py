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

from dataclasses import dataclass
from typing import Any

import numpy as np
from einops import repeat

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin


@dataclass
class AddVariateIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add variate_id to data_entry
    """

    fields: tuple[str, ...]
    max_dim: int
    optional_fields: tuple[str, ...] = tuple()
    variate_id_field: str = "variate_id"
    expected_ndim: int = 2
    randomize: bool = False
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # print(f"[wj debug] AddVariateIndex, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] fields: {self.fields}")
        # print(f"[wj debug] optional_fields: {self.optional_fields}")
        # print(f"[wj debug] variate_id_field: {self.variate_id_field}")
        # print(f"[wj debug] expected_ndim: {self.expected_ndim}")
        # print(f"[wj debug] randomize: {self.randomize}")
        # print(f"[wj debug] collection_type: {self.collection_type}")
        # print(f"[wj debug] max_dim: {self.max_dim}")
        self.counter = 0
        self.dimensions = (
            np.random.choice(self.max_dim, size=self.max_dim, replace=False)
            if self.randomize
            else list(range(self.max_dim))
        )
        # print(f"[wj debug] dimensions: {self.dimensions}")
        data_entry[self.variate_id_field] = self.collect_func(
            self._generate_variate_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_variate_id(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        # print(f"[wj debug] _generate_variate_id, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] field: {field}")
        arr = data_entry[field]
        # print(f"[wj debug] arr shape: {arr.shape}")
        self.check_ndim(field, arr, self.expected_ndim)
        # print(f"[wj debug] arr: {len(arr)}, arr shape: {arr.shape}")
        # print(f"[wj debug] arr: {len(arr)}, arr shape: {arr[0].shape}")
        """
        [wj debug] arr: 1, arr shape: (1, 5, 128)
        """
        dim, time = arr.shape[:2]
        
        if self.counter + dim > self.max_dim:
            raise ValueError(
                f"Variate ({self.counter + dim}) exceeds maximum variate {self.max_dim}. "
            )
        field_dim_id = repeat(
            np.asarray(self.dimensions[self.counter : self.counter + dim], dtype=int),
            "var -> var time",
            time=time,
        )
        # print(f"[wj debug] field_dim_id shape: {field_dim_id.shape}")
        # print(f"[wj debug] self.counter: {self.counter}")
        self.counter += dim
        return field_dim_id


@dataclass
class AddTimeIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add time_id to data_entry
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    time_id_field: str = "time_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        add sequence_id
        """
        # print(f"[wj debug] AddTimeIndex, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] fields: {self.fields}")
        # print(f"[wj debug] optional_fields: {self.optional_fields}")
        # print(f"[wj debug] time_id_field: {self.time_id_field}")
        # print(f"[wj debug] expected_ndim: {self.expected_ndim}")
        # print(f"[wj debug] collection_type: {self.collection_type}")
        data_entry[self.time_id_field] = self.collect_func(
            self._generate_time_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_time_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        # print(f"[wj debug] _generate_time_id, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] field: {field}")
        arr = data_entry[field]
        # print(f"[wj debug] arr: {arr.shape}")
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_seq_id = np.arange(time)
        # print(f"[wj debug] field_seq_id: {field_seq_id.shape}")
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        # print(f"[wj debug] field_seq_id: {field_seq_id.shape}")
        return field_seq_id


@dataclass
class AddObservedMask(CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    observed_mask_field: str = "observed_mask"
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # print(f"[wj debug] AddObservedMask, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] fields: {self.fields}")
        # print(f"[wj debug] optional_fields: {self.optional_fields}")
        # print(f"[wj debug] observed_mask_field: {self.observed_mask_field}")
        # print(f"[wj debug] collection_type: {self.collection_type}")
        observed_mask = self.collect_func(
            self._generate_observed_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.observed_mask_field] = observed_mask
        return data_entry

    @staticmethod
    def _generate_observed_mask(data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        return ~np.isnan(arr)
