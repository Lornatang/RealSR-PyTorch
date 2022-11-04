# Usage

## Step1: Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Step2: Prepare the dataset in the following format

```text
# Dataset struct
- RealSR_V3
    - Nikon
        - train
            - x4
                - HR
                    - Nikon_001.png
                    - ...
                - LR
                    - Nikon_001.png
                    - ...
        - test
            - x4
                - HR
                    - Nikon_001.png
                    - ...
                - LR
                    - Nikon_001.png
                    - ...
```

## Step3: Preprocess the train dataset

```bash
cd <RealSR-PyTorch-main>/scripts
python3 run.py
```

## Step4: Check that the final dataset directory schema is completely correct

```text
- RealSR_V3
    - train
    - test
```
