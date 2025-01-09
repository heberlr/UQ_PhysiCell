#!/bin/bash

# Define the target directory
TARGET_DIR="./examples"

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR

# Navigate to the target directory
cd $TARGET_DIR

# Download the PhysiCell zip file
curl -L -o PhysiCell.zip https://github.com/MathCancer/PhysiCell/archive/refs/heads/master.zip

# Unzip the downloaded file - this will create a new directory called PhysiCell-master
unzip PhysiCell.zip

# Remove the zip file
rm PhysiCell.zip

echo "PhysiCell has been downloaded into the examples folder."