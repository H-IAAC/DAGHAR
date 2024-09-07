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


## Processing Datasets




## Extending DAGHAR


## Citation

Oliveira Napoli, O., Duarte, D., Alves, P., Hubert Palo Soto, D., Evangelista de Oliveira, H., Rocha, A., Boccato, L., & Borin, E. (2024). DAGHAR: A Benchmark for Domain Adaptation and Generalization in Smartphone-Based Human Activity Recognition [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11992126