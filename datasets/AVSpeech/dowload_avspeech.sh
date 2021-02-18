#!/bin/bash

if [ -d train ]
then 
	echo "Training directory already exists. Would you like to erase it or keep its contents? (erase/keep):"
	read train_choice
	while [ ! -z $train_choice ]
	do
	if [ $train_choice == "erase" ]
	then 
		rm -rf train
		mkdir train
		echo "Erased all previous training files and created a new empty training directory."
		break
	elif [ $train_choice == "keep" ]
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

# Download the csv files with all the video ids


#TRAIN_CSV=id_files/avspeech_train.csv
#while IFS=, read -r id start end x y
#do
#	echo $id
#done < $TRAIN_CSV
