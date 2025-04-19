# 🚗💨AI-Solutions-for-intelligent-vehicles

## 📥 Importing the dataset needed for the project:
For developing the code I worked with PVS 1 from the linked dataset below in order to keep development smooth and working with smaller dataset for the models first before later explanding to the full dataset.

## 🧮 Dataset:
the dataset used for this project is the PVS dataset from the following link:
https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets

From this, the following files were used and will be used for the other PVS datasets:
- dataset_lables.csv
- dataset_gps.csv
- dataset_mpu_left.csv
- dataset_mpu_right.csv
- dataset_gps_mpu_left.csv
- dataset_gps_mpu_right.csv

## 🔄 Pre processing:
### Missing Values:
__Checking:__
- gps data:
    - ageofdgpsdata = 1467
    - dgpsid = 1467
    - activity = 1467
    - annotation = 1467
- mpu data left = none
- mpu data right = none
- combined gps and mpu data left = none
- combined gps and mpu data right = none

__Handling:__
The values missing in data_gps (1467 values in each column) means that for those columns all the values are missing, since they don't offer contribution to the dataset, they can be safely removed without causing any issues.

## 🤖 Models:

## 🤝 Main:


## 📂 File structure:
1. **preprocessing.py**: This file contains the code for pre-processing the dataset.
2. **model.py**: This file contains the code for training the models and making predictions.
3. **main.py**: This file contains the main function that runs the code.

## How to run the code:
1. Clone the repository to your local machine.
2. Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
3. Run the main script:
```bash
python main.py
```
