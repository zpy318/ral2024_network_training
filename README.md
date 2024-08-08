# Installation
We recommend running this in a virtual environment:
```
# generate a virtual environment with name test_env and Python 3.8.16 installed
conda create -n test_env python=3.8.16
# activate the environment
conda activate test_env
# deactivate the environment
conda deactivate
# delete the virtual environment and all its packages
conda remove -n test_env --all
```
To install all necessary packages, please navigate to the cloned directory and run the following code in the terminal:
```
pip install -r requirements.txt
```

# Usage
Please run the corresponding python script to train the network.
```
python3 training_B_scan_segmentation_input_1024_512_no_crop.py
python3 training_B_scan_tool_tip_and_base_prediction_input_1024_512_no_crop.py
python3 training_camera_tool_tip_and_base_prediction_input_600_800_crop_to_480_640.py
```
