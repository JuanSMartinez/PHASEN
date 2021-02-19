#!/bin/bash

Download_Training_Set(){
	# The CSV files with video ids must have been downloaded in advance
	# Downloading the files via wget or curl proved to be extremely difficult (requires Google authentication) and not worth the effort

	TRAIN_CSV="avspeech_train.csv"
	TEST_CSV="avspeech_test.csv"
	PREFIX="http://youtube.com/watch?v="

	SUCCESS_TRAIN=0
	FAIL_TRAIN=0
	TOTAL_TRAIN=0
	if [ -f "$TRAIN_CSV" ]
	then
		while IFS=, read -r ID START END x y
		do
			((TOTAL_TRAIN = TOTAL_TRAIN + 1))
			# Try to download and get the speech of a video	
			OUTPUT="train/_$ID.%(ext)s"
			URL="$PREFIX$ID"
			if youtube-dl -x --audio-format "wav" --audio-quality 0 -o "$OUTPUT" "$URL"
			then
				ffmpeg -nostdin -hide_banner -loglevel error -i "train/_$ID.wav" -ss "$START" -to "$END" -c copy "train/$ID.wav"
				rm "train/_$ID.wav"
				((SUCCESS_TRAIN = SUCCESS_TRAIN + 1))
			else
				((FAIL_TRAIN = FAIL_TRAIN + 1))
			fi
		done < "$TRAIN_CSV"
		echo "Total Files: $TOTAL_TRAIN. Succesfully downloaded and processed: $SUCCESS_TRAIN. Failed to download: $FAIL_TRAIN" > log_train.txt
	else
		echo "ERROR: Metadata for training files not found. Download the file 'avspeech_train.csv', place it along this script and run again."
		exit -1
	fi
}


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
		Download_Training_Set
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


