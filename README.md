# DAGHAR

This is the official repository for the DAGHAR dataset, available at [Zenodo](https://zenodo.org/records/11992126). It contains scripts to read, process, and standardize datasets in order to generate DAGHAR dataset. It also allows extending the DAGHAR dataset with new datasets, using the same processing and standardization steps.

## Quick Run

To generate the `raw_balanced` and `standartized_balanced` views of the DAGHAR dataset, you must:

1. Clone this repository;
2. Download the original datasets using the `download_original_data.sh` script;
3. Install the required packages using the `requirements.txt` file;
4. Run the `dataset_generator.py` script.

This can be done using the following commands (in a Unix-like system):

```bash
git clone https://github.com/H-IAAC/DAGHAR.git DAGHAR
cd DAGHAR
./download_original_data.sh
pip install -r requirements.txt
python dataset_generator.py
```

This will generate the `raw_balanced` and `standartized_balanced` views of the DAGHAR dataset in the `data/views` directory.
Below we provide more details on how to extend the DAGHAR dataset with new datasets.

## Downloading the Original Datasets

The repository is organized as follows:

```
DAGHAR/
├── data/
│   ├── original/
│   │   ├── KuHar/
│   │   ├── MotionSense/
│   │   ├── RealWord/
│   │   ├── UCI/
│   │   └── WISDM/
├── steps.py
├── pipelines.py
├── readers.py
├── dataset_generator.py
├── requirements.txt
├── download_original_data.sh
├── README.md
└── LICENSE
```

The `data` directory contains the original datasets, as they are downloaded from their respective sources. 

In order to generate the DAGHAR views, you must download the original datasets, decompress them, and place them in the `data/original` directory, inside the respective dataset directory.
To facilitate this process, we provide a shell script named `download_original_data.sh`. This script downloads the datasets and decompresses them in the correct directory. 
To use it, run the following command:

```bash
./download_original_data.sh
```
> **NOTE**: The user must have `wget` and `unzip` installed in the system to run the script. In Ubuntu, you can install them using the following command:
> ```bash
> sudo apt-get install wget unzip
> ```

The script will download the datasets in the `data/original` directory. It will not download the datasets that are already present in the directory. If you want to download them again, you must delete the respective directories or files.


We used the following datasets:

- **Ku-HAR**, from "Sikder, N. and Nahid, A.A., 2021. KU-HAR: An open dataset for heterogeneous human activity recognition. Pattern Recognition Letters, 146, pp.46-54". Distributed under CC BY 4.0. [Download link](https://data.mendeley.com/datasets/45f952y38r/5), the `1.Raw_time_domian_data.zip` file.
- **MotionSense**, from "Malekzadeh, M., Clegg, R.G., Cavallaro, A. and Haddadi, H., 2019, April. Mobile sensor data anonymization. In Proceedings of the international conference on internet of things design and implementation (pp. 49-58)". Distributed under Open Data Commons Open Database License (ODbL) v1.0. [Download link](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset), the `A_DeviceMotion_data.zip` file.
- **RealWorld**, from "Sztyler, T. and Stuckenschmidt, H., 2016, March. On-body localization of wearable devices: An investigation of position-aware activity recognition. In 2016 IEEE international conference on pervasive computing and communications (PerCom) (pp. 1-9). IEEE" link. We obtained explicitly permission to distribute a copy of the preprocessed data from the original authors. [Download link](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)
- **UCI-HAR**, from "Reyes-Ortiz, J.L., Oneto, L., Samà, A., Parra, X. and Anguita, D., 2016. Transition-aware human activity recognition using smartphones. Neurocomputing, 171, pp.754-767". Distributed under CC BY 4.0. [Download link](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)
- **WISDM**, from "Weiss, G.M., Yoneda, K. and Hayajneh, T., 2019. Smartphone and smartwatch-based biometrics using activities of daily living. Ieee Access, 7, pp.133190-133202". Distributed under CC BY 4.0. [Download link](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)


## Generate the DAGHAR Datasets

### Pre-requisites

To generate the DAGHAR datasets, you must have python 3.8 or higher installed in your system. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Generate the Datasets

To generate the `raw_balanced` and `standartized_balanced` views from original datasets, we provide a python script named `dataset_generator.py`. 
You can run the script using the following command:

```bash
python dataset_generator.py
```

This file generates the `raw` and `standartized` views of the datasets at the `data/processed` directory. This process may take a while, depending on the dataset size and the number of views to be generated.


## Extending the DAGHAR Dataset

The standardization process is a pipeline that comprises several steps to standardize the datasets. The pipeline is dataset-specific, as each dataset has its own standardization process.
Thus, scripts is mainly divided in two parts: read the dataset and standardize the dataset.
The first part is dataset-specific, as it reads the dataset and generates a pandas dataframe with all the data. The second part is most dataset-agnostic, as it standardizes the dataset using a pipeline of operations. We provide a set of operators that are used in the standardization process, such as resampling, windowing, and adding labels. We also encourage the user to create new operators that may be useful in the standardization process and to reuse the operators in other datasets.

### Reading the Dataset

Reading the dataset is a dataset-specific process. It involves reading the CSV files, concatenating them, and generating a **single pandas dataframe** with all the data. 
The resultant dataframe has a row for each **instant of capture** (and not a window). The columns of the dataframe are dataset-specific, but they, at least, must contain the following columns (required ones):
- Accelerometer x-axis (`accel-x` column); y-axis (`accel-y` column); z-axis (`accel-z` column);
- Gyroscope x-axis (`gyro-x` column); y-axis (`gyro-y` column); z-axis (`gyro-z` column);
- Timestamp from the accelerometer (`accel-start-time` column) and gyroscope (`gyro-strrt-time` column);
- Label (`activity code` column);
- The user id (`user` column);
- The trial id (`serial` column), that is the trial associated with the time instant, as one user may have multiple trials;
- The index of the time instant within a user's trial (`index` column);
- The file that the row was extracted from (`csv` column);

The dataframe may also contain other metadata information (additional columns, along the required ones), which is dataset-specific and can be discarded (or used) during the standardization process.


All functions that read the datasets are defined in the `readers.py` file. 
Each reader is a function that receives the dataset directory path and returns a pandas dataframe.
Thus, in order to extend the DAGHAR dataset with a new dataset, you must create a new reader function in the `readers.py` file.


### Creating Pipelines

As datasets have different formats, the standardization process is dataset-specific. 
For instance, the standardization process for the `KuHar` dataset involves resampling the data to 20Hz and creating 3-second windows, while the standardization process for the `WISDM` dataset involves passing a butterworth filter, resampling the data to 20Hz, and creating 3-second windows.
Thus, each dataset has its own sequence of steps which we call `Pipeline`.

The steps are simple operations such as resampling, windowing, and adding labels. Each step is a class, in `steps.py` file, that implements the `__call__` method, which receives a pandas dataframe and returns a pandas dataframe. The `__call__` method is the operator's implementation.
The pipeline is composed of a sequence of steps an  is defined in the `pipeline.py` file.

To extend the DAGHAR dataset with a new dataset, you must create a new pipeline and add it to the `pipelines.py` file, under the `pipelines` dictionary. The key of the dictionary is the dataset name, and the value is the pipeline object. If you need other steps then the ones provided, you must create new operators and add them to the `steps.py` file.

> **NOTE** The order of the operators is important, as some operators may require columns that may be added from other operators (which must run before).

### Standardizing the Dataset

The main standardization process is done in the `dataset_generator.py` file, which standardizes the dataset using the pipeline defined in the `pipelines.py` file.
In fact, this file is the main entrypoint for user interaction, as it generates the `raw_balanced` and `standartized_balanced` views of the dataset.
It will responsible to run over all the datasets and pipelines, generating the views.


#### Interactive Example

The following code snippet illustrates a fictitious standardization process for the `KuHar` dataset that resamples it to 20Hz and creates 3-second windows (**using this kind of code and operators is not a rule**):

```python

def read_kuhar(kuhar_dir_path: str) -> pd.DataFrame:
    # This is dataset-specific code. It reads the CSV files and generates a single dataframe with all the data.
    ... 
    return dataframe

# -----------------------------------------------------------------------------
# 1. Load the datasets and generate a single pandas dataframe with all the data
# -----------------------------------------------------------------------------

dataframe = read_kuhar("data/original/KuHar/1.Raw_time_domian_data")

# -----------------------------------------------------------------------------
# 2. Create operaators
# -----------------------------------------------------------------------------

# List with columns that are features
feature_columns = [
    "accel-x",
    "accel-y",
    "accel-z",
    "gyro-x",
    "gyro-y",
    "gyro-z",
]


# Instantiate the object that resample the data to 20Hz
# (supose that the original dataset has a constant sample rate equal to 100Hz)
resampler = ResamplerPoly(
    features_to_select=feature_columns, # Name of the columns that will be used
                                        # as features.
    up=2,                               # The upsampling factor.
    down=10,                            # The downsampling factor.
    groupby_column="csv",               # Group by csv column. 
                                        # Resampling is done for each group of
                                        # csv column group.
)

# Instantiate the object taht creates the windows
windowizer = Windowize(
    features_to_select=feature_columns, # Name of the coluns that will be used
                                        # as features.
    samples_per_window=60,              # Number of samples per window.
    samples_per_overlap=0,              # Number of samples that overlap.
    groupby_column="csv",               # Group by csv column.
                                        # Resampling is done for each group of
                                        # csv column group.
)

# -----------------------------------------------------------------------------
# 3. Create the pipeline object, passing the operator object list as parameters
# -----------------------------------------------------------------------------

# Create the pipeline
# 1. Resample the data
# 2. Create the windows
pipeline = Pipeline(
    [
        resampler,
        windowizer,
    ]
)

# -----------------------------------------------------------------------------
# 4. Execute the pipeline, passing the dataframe as a parameter
# -----------------------------------------------------------------------------

standartized_dataset = pipeline(dataframe)
```

The pipeline operators are usually shared with other datasets, as they are generic.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Citation

Oliveira Napoli, O., Duarte, D., Alves, P., Hubert Palo Soto, D., Evangelista de Oliveira, H., Rocha, A., Boccato, L., & Borin, E. (2024). DAGHAR: A Benchmark for Domain Adaptation and Generalization in Smartphone-Based Human Activity Recognition [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11992126