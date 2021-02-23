#!/bin/bash

Download_Data_Set(){

	# For this dataset, we can download the CSV files from the script
	
	SET=$1
	echo "Downloading metadata file ..."
	if [ "$SET" == "train" ]
	then
		# The paper declares to have used the 'Balanced train' set. But this set only has ~22k videos. We use the 'Unbalanced train' set here.
		curl -LO http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv	
		cat unbalanced_train_segments.csv | sed 1,3d > metadata.csv
		rm unbalanced_train_segments.csv
		SIZE=10
	else
		# For testing, we use the 'Eval_segments' set
		curl -LO http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
		cat eval_segments.csv | sed 1,3d > metadata.csv
		rm eval_segments.csv
		SIZE=2
	fi

	CSV_FILE="metadata.csv"
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
				OUTPUT="$SET/_$ID.%(ext)s"
				URL="$PREFIX$ID"
				if youtube-dl -x --audio-format "wav" --audio-quality 0 -o "$OUTPUT" "$URL"
				then
					ffmpeg -nostdin -hide_banner -loglevel error -i "$SET/_$ID.wav" -ss "$START" -to "$END" -c copy "$SET/$ID.wav"
					rm "$SET/_$ID.wav"
					((SUCCESS = SUCCESS + 1))
				fi
			else
				break
			fi
		done
		echo "Complete! $SUCCESS files were downloaded and preprocessed in the '$SET' directory."
		rm "$CSV_FILE"	
	else
		echo "ERROR: Metadata for data files not found. Try to run this script again or download CSV files manually." 
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


