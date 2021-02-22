#!/bin/bash

Download_Data_Set(){
	# The CSV files with video ids must have been downloaded in advance
	# Downloading the files via wget or curl proved to be difficult (requires Google authentication) and not worth the effort
	
	SET=$1
	if [ "$SET" == "train" ]
	then
		CSV_FILE="avspeech_train.csv"
		DEST_DIR="train/"
		LOG="log_train.txt"
	else
		CSV_FILE="avspeech_test.csv"
		DEST_DIR="test/"
		LOG="log_test.txt"
	fi
	PREFIX="http://youtube.com/watch?v="
	
	SUCCESS=0
	FAIL=0
	TOTAL=0
	if [ -f "$CSV_FILE" ]
	then
		echo "Processing metadata of video id's for the $SET dataset. This might take a while ..."
		VIDS=( $( cut -d ',' -f1 $CSV_FILE ) )
		START_STAMPS=( $( cut -d ',' -f2 $CSV_FILE ) )
		END_STAMPS=( $( cut -d ',' -f3 $CSV_FILE ) )
		TOTAL_VIDS=( ${#VIDS[@]} )
		if [ uname == "Darwin" ]
		then
			# Using Mac OS
			IDX=( $( seq 0 1 $((TOTAL_VIDS - 1)) | gshuf ) )
		else
			IDX=( $( seq 0 1 $((TOTAL_VIDS - 1)) | shuf ) )
		fi
		
		for i in "${IDX[@]}"
		do
			#TODO
		done

		while IFS=, read -r ID START END x y
		do
			((TOTAL = TOTAL + 1))
			# Try to download and get the speech of a video	
			OUTPUT="$DEST_DIR/_$ID.%(ext)s"
			URL="$PREFIX$ID"
			if youtube-dl -x --audio-format "wav" --audio-quality 0 -o "$OUTPUT" "$URL"
			then
				ffmpeg -nostdin -hide_banner -loglevel error -i "$DEST_DIR/_$ID.wav" -ss "$START" -to "$END" -c copy "$DEST_DIR/$ID.wav"
				rm "$DEST_DIR/_$ID.wav"
				((SUCCESS = SUCCESS + 1))
			else
				((FAIL = FAIL + 1))
			fi
		done < "$CSV_FILE"
		echo "Total Files: $TOTAL. Succesfully downloaded and processed: $SUCCESS. Failed to download: $FAIL" > "$LOG"
	else
		echo "ERROR: Metadata for data files not found. Download the files 'avspeech_train.csv' and 'avspeech_test.csv' and place them along this script. Then run again."
		exit -2
	fi
	echo "PROCESSED FINISHED. CHECK LOG FOR DETAILS."
}

echo "Which dataset would you like to download? (train/test)?"
read DATASET
if [ ! "$DATASET" == "train" ] && [ ! "$DATASET" == "test" ]
then
	echo "ERROR: Invalid Choice"
	exit -1
fi

if [ -d "$DATASET" ]
then 
	echo "Dataset directory already exists. Would you like to erase it or keep its contents? (erase/keep):"
	read CHOICE
	while [ ! -z "$CHOICE" ]
	do
	if [ "$CHOICE" == "erase" ]
	then 
		rm -rf "$DATASET"
		mkdir "$DATASET"
		echo "Erased all previous files and created a new empty directory."
		Download_Data_Set "$DATASET"
		break
	elif [ "$CHOICE" == "keep" ]
	then
		echo "Keeping files inside the directory."
		break
	else
		echo "Invalid choice. Please choose to erase or keep:"
		read CHOICE 
	fi
	done
else
	mkdir "$DATASET"
	echo "Created empty directory."
	Download_Data_Set "$DATASET"
fi


