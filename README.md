# DAGHAR

This is the official repository for the DAGHAR dataset, available at [Zenodo](https://zenodo.org/records/11992126). It contains scripts to read, process, and standardize datasets in order to generate DAGHAR dataset. It also allows extending the DAGHAR dataset with new datasets, using the same processing and standardization steps.

## Directory Structure

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
├── processing_scripts/
│   ├── dataset_generator.py
│   ├── dataset_processor.py
│   └── dataset_standardizer.py
├── README.md
└── LICENSE
```

The `data` directory contains the original datasets, as they are downloaded from their respective sources. The `processing_scripts` directory contains the scripts used to process and standardize the datasets, generating the `standartized_balanced` and `raw_balanced` directories, that is, the standardized and raw balanced views of DAGHAR, respectively.


### Downloading the Original Datasets

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



## Dataset Generation Process

To generate the datasets, you can run the `dataset_generator.py` and `inter_dataset_balancer.py` files, which is in this directory. This file generates the `raw` and `standartized` views that can be unbalanced, intra_balanced (balanced per activity), and inter_balanced (balanced per activity and user), that is, all users have the same number of samples per activity. For generate inter_balanced views,  that balances each dataset based the minimum number of classes per split (train, validation or test), of all datasets, you must run the file `inter_dataset_balancer.py`.

The order is very important, to run `inter_dataset_balancer.py` you need the intra_balanced datasets, so if you don't have these views on your machine, you should first run `dataset_generator.py` file and then run the `inter_dataset_balancer.py` file.

## Standardization Process

The standardization process also called the standardization pipeline, comprises the execution of several steps (operations) per dataset.
Each dataset has its standardization pipeline, as they are different from each other. 
The operators are all defined in the `dataset_processor.py` file. The operators are defined as classes, and each class has a `__call__` method, which receives a pandas dataframe and returns a pandas dataframe. The `__call__` method is the operator's implementation.

---

**NOTE**

- The order of the operators is important, as some operators may require columns that may be added from other operators (which must run before).
- Seldom, some operators may return multiple pandas Dataframes. 

---

The standardization codes from `dataset_generator.py` file usually comprise the following steps (**this is not a strict rule**):

1. Load the datasets and generate a single pandas dataframe with all the data, where each row represents a single instant of capture (and now a window). The loading is a dataset-specific process. The dataframe generated **usually** have the following columns (**this is not a rule**):
- A column for the x-axis acceleration (`accel-x` column); y-axis acceleration (`accel-y` column); z-axis acceleration (`accel-z` column); gyroscope x-axis (`gyro-x` column); gyroscope y-axis (`gyro-y` column); gyroscope z-axis (`gyro-z` column); and the timestamp from the accelerometer (`accel-timestamp`) and gyroscope (`gyro-timestamp`), if provided.
- A column for the label (`activity code` column).
- A column for the user id (`user` column), if provided.
- A column for x-axis gravity (`gravity-x` column); y-axis gravity (`gravity-y` column); and z-axis gravity (`gravity-z` column) if provided.
- A serial column, which represents the attempt that the collection was made (`serial` column), if provided. For instance, if the user has a time series running in the morning and another in the afternoon, it will be two different serial numbers.
- A CSV or file column, which represents the file that the row was extracted from (`csv` column). 
- An index column, that is, a column that represents the row index from the CSV file (`index` column).
- Any other column that may be useful for the standardization process or metadata.
2. Create the operator objects.
3. Crete the pipeline object, passing the operator object list as parameters.
4. Execute the pipeline, passing the dataframe as a parameter.

The following code snippet illustrates a fictitious standardization process for the `KuHar` dataset that resamples it to 20Hz and creates 3-second windows (**using this kind of code and operators is not a rule**):

```python

def read_kuhar(kuhar_dir_path: str) -> pd.DataFrame:
    # This is dataset-specific code. It reads the CSV files and generates a single dataframe with all the data.
    ... 
    return dataframe

# -----------------------------------------------------------------------------
# 1. Load the datasets and generate a single pandas dataframe with all the data
# -----------------------------------------------------------------------------

dataframe = read_kuhar("../data/original/KuHar/1.Raw_time_domian_data")

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
# 3. Add a column wit activity code
pipeline = Pipeline(
    [
        differ,
        resampler,
        windowizer,
        standard_label_adder
    ]
)

# -----------------------------------------------------------------------------
# 4. Execute the pipeline, passing the dataframe as a parameter
# -----------------------------------------------------------------------------

standartized_dataset = pipeline(dataframe)
```

The example above is to read and preprocess a dataset, in the `dataset_generator.py` file there a dictionary with a specific pipeline for each dataset and view. 

The pipeline operators are usually shared with other datasets, as they are generic.

## Extending the DAGHAR Dataset

....


## Citation

Oliveira Napoli, O., Duarte, D., Alves, P., Hubert Palo Soto, D., Evangelista de Oliveira, H., Rocha, A., Boccato, L., & Borin, E. (2024). DAGHAR: A Benchmark for Domain Adaptation and Generalization in Smartphone-Based Human Activity Recognition [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11992126