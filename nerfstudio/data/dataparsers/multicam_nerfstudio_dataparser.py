# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations


from pathlib import Path
from PIL import Image
from typing import Literal, Optional, Tuple, Type
from dataclasses import dataclass, field
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio,NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class MulticamNerfstudioDataParserConfig(NerfstudioDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: MulticamNerfstudio)
    includes_timestamp: bool = True



MAX_AUTO_RESOLUTION = 1600
@dataclass
class MulticamNerfstudio(Nerfstudio):
    """Nerfstudio DatasetParser"""
    config: MulticamNerfstudioDataParserConfig
    includes_time: bool = True
    downscale_factor: Optional[int] = 4
    """Whether to include timestamps"""

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """


        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            parts = Path(filepath).parts
            new_parts = parts[:-2] + (f"{downsample_folder_prefix}{self.downscale_factor}",) + parts[-1:]
            new_downscale_path = Path(data_dir) / Path(*new_parts)
            return  new_downscale_path

        return data_dir / filepath


