# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/train/x4/HR --output_dir ../data/RealSR_V3/train/x4/HR --image_size 228 --step 228 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/train/x3/HR --output_dir ../data/RealSR_V3/train/x3/HR --image_size 228 --step 228 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/train/x2/HR --output_dir ../data/RealSR_V3/train/x2/HR --image_size 228 --step 228 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/train/x4/LR --output_dir ../data/RealSR_V3/train/x4/LR --image_size 57 --step 57 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/train/x3/LR --output_dir ../data/RealSR_V3/train/x3/LR --image_size 76 --step 76 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/train/x2/LR --output_dir ../data/RealSR_V3/train/x2/LR --image_size 114 --step 114 --num_workers 16")

os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/test/x4/HR --output_dir ../data/RealSR_V3/test/x4/HR --image_size 228 --step 228 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/test/x3/HR --output_dir ../data/RealSR_V3/test/x3/HR --image_size 228 --step 228 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/test/x2/HR --output_dir ../data/RealSR_V3/test/x2/HR --image_size 228 --step 228 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/test/x4/LR --output_dir ../data/RealSR_V3/test/x4/LR --image_size 57 --step 57 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/test/x3/LR --output_dir ../data/RealSR_V3/test/x3/LR --image_size 76 --step 76 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/Nikon/test/x2/LR --output_dir ../data/RealSR_V3/test/x2/LR --image_size 114 --step 114 --num_workers 16")
