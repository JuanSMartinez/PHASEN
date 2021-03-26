#!/bin/bash

Download_Data_Set(){
	# The CSV files with video ids must have been downloaded in advance
	# Downloading the files via wget or curl proved to be difficult (requires Google authentication) and not worth the effort

	SET=$1
	if [ "$SET" == "train" ]
	then
		CSV_FILE="avspeech_train.csv"
		SIZE=5
	else
		CSV_FILE="avspeech_test.csv"
		SIZE=5
	fi
	PREFIX="http://youtube.com/watch?v="

	SUCCESS=0
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
			IDX=( $( seq -f %1.0f 0 1 $((TOTAL_VIDS - 1)) | gshuf ) )
		else
			# Using Linux
			IDX=( $( seq -f %1.0f 0 1 $((TOTAL_VIDS - 1)) | shuf ) )
		fi
		echo "Beginning download and processing of $SIZE videos, this will take even more time ..."
		for i in "${IDX[@]}"
		do
			if [ $SUCCESS -lt $SIZE ]
			then
				ID="${VIDS[$i]}"
				START="${START_STAMPS[$i]}"
				END="${END_STAMPS[$i]}"
				OUTPUT="$SET/temporal_video_$ID.%(ext)s"
				URL="$PREFIX$ID"
				if youtube-dl -x --audio-format "wav" --audio-quality 0 -o "$OUTPUT" "$URL"
				then
					if ffmpeg -nostdin -hide_banner -loglevel error -i "$SET/temporal_video_$ID.wav" -ss "$START" -to "$END" -c copy "$SET/$ID.wav"
					then
						rm "$SET/temporal_video_$ID.wav"
						((SUCCESS = SUCCESS + 1))
					fi
				fi
			else
				break
			fi
		done
		# cleanup
		find "$SET" -type f ! -name "*.wav" -delete
		COMPLETE=( $( ls "$SET" | wc -l ) )
		echo "Complete. $COMPLETE files were downloaded and preprocessed in the '$SET' directory."
	else
		echo "ERROR: Metadata for data files not found. Download the files 'avspeech_train.csv' and 'avspeech_test.csv' and place them along this script. Then run again."
		exit -2
	fi
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
