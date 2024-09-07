#!/bin/bash


ORIGINAL_DATA_DIR="./data/original"

echo "-------------------------------------------------------------------------"
echo "This script will download and extract the original data DAGHAR dataset."
echo "The data will be saved in the directory: $ORIGINAL_DATA_DIR"
echo "-------------------------------------------------------------------------"
echo ""

die () {
    echo >&2 "$@"
    exit 1
}

download_and_extract_dataset () {
    # $1: download URL
    # $2: dataset name
    DOWNLOAD_URL="$1"
    DATASET_NAME="$2"

    if [ -d "$ORIGINAL_DATA_DIR/$DATASET_NAME" ]; then
        echo "The data for the $DATASET_NAME dataset is already downloaded and extracted."
    else
        echo "Downloading the $DATASET_NAME dataset..."
        wget -O "$ORIGINAL_DATA_DIR/$DATASET_NAME.zip" "$DOWNLOAD_URL" || return 1
        unzip "$ORIGINAL_DATA_DIR/$DATASET_NAME.zip" -d "$ORIGINAL_DATA_DIR/$DATASET_NAME" || return 1
        rm "$ORIGINAL_DATA_DIR/$DATASET_NAME.zip" || return 1
    fi
    return 0
}

################################################################################
# KUHAR
################################################################################

download_and_extract_dataset "https://data.mendeley.com/public-files/datasets/45f952y38r/files/d3126562-b795-4eba-8559-310a25859cc7/file_downloaded" "KuHar" || die "Failed to download and process the KuHar dataset."
echo "-------------------------------------------------------------------------"

################################################################################
# MotionSense
################################################################################

echo "We are downloading the MotionSense dataset from github as it does not requires authentication."
download_and_extract_dataset "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip" "MotionSense" || die "Failed to download and process the MotionSense dataset."
rm -rf "$ORIGINAL_DATA_DIR/MotionSense/__MACOSX" 
echo "-------------------------------------------------------------------------"

################################################################################
# RealWorld
################################################################################

download_and_extract_dataset "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip" "RealWorld" || die "Failed to download and process the RealWorld dataset."

# reorganize the RealWorld dataset
if [ ! -d "$ORIGINAL_DATA_DIR/RealWorld/realworld2016_dataset" ]; then
    mv "$ORIGINAL_DATA_DIR/RealWorld/" "$ORIGINAL_DATA_DIR/temp"
    mkdir -p "$ORIGINAL_DATA_DIR/RealWorld/realworld2016_dataset"
    mv $ORIGINAL_DATA_DIR/temp/* "$ORIGINAL_DATA_DIR/RealWorld/realworld2016_dataset"
    rm -r "$ORIGINAL_DATA_DIR/temp"
fi

echo "-------------------------------------------------------------------------"

################################################################################
# UCI-HAR
################################################################################

download_and_extract_dataset "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip" "UCI" || die "Failed to download and process the UCI dataset."
echo "-------------------------------------------------------------------------"

################################################################################
# WISDM
################################################################################

download_and_extract_dataset "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip" "WISDM" || die "Failed to download and process the WISDM dataset."
echo "-------------------------------------------------------------------------"

if [ ! -d "$ORIGINAL_DATA_DIR/WISDM/wisdm-dataset" ]; then
    unzip "$ORIGINAL_DATA_DIR/WISDM/wisdm-dataset.zip" -d "$ORIGINAL_DATA_DIR/WISDM" || die "Failed to extract the WISDM dataset."
    rm "$ORIGINAL_DATA_DIR/WISDM/wisdm-dataset.zip" || die "Failed to remove the WISDM dataset zip file."
fi

################################################################################
echo "All datasets are downloaded and extracted successfully."