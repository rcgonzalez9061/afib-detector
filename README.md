# afib-detector
An ML based arrhythmia  detector. See my full report [here](https://rcgonzalez9061.github.io/afib-detection-blog/).

## Structure
```
├── assets          Folder to store visualizations and other possible assets
├── data            Data Folder
│   ├── cleaned         Label mappings
│   ├── physionet       MIT BIH data
│   └── temp            Temp folder for when creating vizualizations
├── models          Storage of model parameters organized by architecture
├── notebooks       Notebooks for various tasks
└── src             Libraries for various tasks
```

---
## Replication
If you'd like to replicate my findings, simply follow these steps.

### 1) Clone this repositiory and install the required packages

```
git clone https://github.com/rcgonzalez9061/afib-detector.git
pip install requirements.txt
```

### 2) Run setup script
I've included a setup script to aid with downloading data and pretrained models as well as building the project structure.

```
usage: python setup.py [-h] [--all] [--download-data] [--download-models] [--build]

arguments:
  -h, --help         show this help message and exit
  --all              Performs all actions below
  --download-data    Loads data from Physionet Database
  --download-models  Downloads pretrained models
  --build            Builds folder structure and label mappings 
  --eval             Evaulates models in models folder
```
  
### 4) Train Models (Optional)
If you'd like to retrain the models, run through the `Training.ipynb` notebook.

### 3) Viewing Results
Metrics of the models can viewed by running through the `Model Evaluation.ipynb` notebook. (You should run this after performing the evaluation function in the setup script.)
