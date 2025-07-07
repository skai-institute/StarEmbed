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
from typing import Any, Optional

import numpy as np
from einops import pack

from ._base import Transformation
from ._mixin import CollectFuncMixin, MapFuncMixin


@dataclass
class SequencifyField(Transformation):
    field: str
    axis: int = 0
    target_field: str = "target"
    target_axis: int = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # print(f"[wj debug] SequencifyField, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] field: {self.field}")
        # print(f"[wj debug] target_field: {self.target_field}")
        # print(f"[wj debug] target_axis: {self.target_axis}")
        # print(f"[wj debug] axis: {self.axis}")
        # print(f"[wj debug] data_entry[self.field] shape: {data_entry[self.field].shape}")
        # print(f"[wj debug] data_entry[self.target_field] shape: {data_entry[self.target_field].shape}")
        data_entry[self.field] = data_entry[self.field].repeat(
            data_entry[self.target_field].shape[self.target_axis], axis=self.axis
        )
        # print(f"[wj debug] data_entry[self.field] shape: {data_entry[self.field].shape}")
        return data_entry


@dataclass
class PackFields(CollectFuncMixin, Transformation):
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* time feat" if self.feat else "* time"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # print(f"[wj debug] PackFields, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] fields: {self.fields}")
        # print(f"[wj debug] output_field: {self.output_field}")
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = pack(fields, self.pack_str)[0]
            # print(f"[wj debug] output_field shape: {output_field.shape}")
            data_entry |= {self.output_field: output_field}
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))


@dataclass
class FlatPackFields(CollectFuncMixin, Transformation):
    output_field: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # print(f"[wj debug] FlatPackFields, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] fields: {self.fields}")
        # print(f"[wj debug] output_field: {self.output_field}")
        fields = self.collect_func_list(
            self.pop_field,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        if len(fields) > 0:
            output_field = pack(fields, self.pack_str)[0]
            # print(f"[wj debug] output_field shape: {output_field.shape}")
            data_entry |= {self.output_field: output_field}
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        return data_entry

    @staticmethod
    def pop_field(data_entry: dict[str, Any], field: str) -> Any:
        return np.asarray(data_entry.pop(field))


@dataclass
class PackCollection(Transformation):
    field: str
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* time feat" if self.feat else "* time"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        collection = data_entry[self.field]
        if isinstance(collection, dict):
            collection = list(collection.values())
        data_entry[self.field] = pack(collection, self.pack_str)[0]
        return data_entry


@dataclass
class FlatPackCollection(Transformation):
    field: str
    feat: bool = False

    def __post_init__(self):
        self.pack_str: str = "* feat" if self.feat else "*"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # print(f"[wj debug] FlatPackCollection, file path: {__file__}")
        # print(f"[wj debug] data_entry: {data_entry.keys()}")
        # print(f"[wj debug] field: {self.field}")
        collection = data_entry[self.field]
        if isinstance(collection, dict):
            collection = list(collection.values())
        data_entry[self.field] = pack(collection, self.pack_str)[0]
        # print(f"[wj debug] data_entry[self.field] shape: {data_entry[self.field].shape}")
        return data_entry


@dataclass
class Transpose(MapFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    axes: Optional[tuple[int, ...]] = None

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.transpose,
            data_entry,
            fields=self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def transpose(self, data_entry: dict[str, Any], field: str) -> Any:
        out = data_entry[field].transpose(self.axes)
        return out
