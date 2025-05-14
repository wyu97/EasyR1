# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Env config
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, List

@dataclass
class EnvConfig:
    bsize: int = 2
    n: int = 1
    avd_name: str = 'AndroidWorldAvd'
    android_avd_home: str = '/root/android/avd'
    emulator_path: str = '/root/android/emulator/emulator'
    adb_path: str = '/root/android/platform-tools/adb'
    max_steps: int = 10
    save_path: str = '/root/output'
    image_size: list = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
