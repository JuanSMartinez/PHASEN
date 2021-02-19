#!/bin/bash

if [ -d train ]
then 
	echo "Training directory already exists. Would you like to erase it or keep its contents? (erase/keep):"
	read train_choice
	while [ ! -z "$train_choice" ]
	do
	if [ "$train_choice" == "erase" ]
	then 
		rm -rf train
		mkdir train
		echo "Erased all previous training files and created a new empty training directory."
		break
	elif [ "$train_choice" == "keep" ]
	then
		echo "Keeping files inside the train directory."
		break
	else
		echo "Invalid choice. Please choose to erase or keep:"
		read train_choice
	fi
	done
else
	mkdir train
	echo "Created empty training directory."
fi

# The CSV files with video ids must have been downloaded in advance
# Downloading the files via wget or curl proved to be extremely difficult (requires Google authentication) and not worth the effort

TRAIN_CSV="avspeech_train.csv"
TEST_CSV="avspeech_test.csv"
if [ -f "$TRAIN_CSV" ]
then
	while IFS=, read -r id start end x y
	do
		echo $id
	done < "$TRAIN_CSV"
else
	echo "ERROR: Metadata for training files not found. Download the file 'avspeech_train.csv', place it along this script and run again."
	exit -1
fi
