#!/bin/bash

CONFIG_DIR="../configs"

for STR in $CONFIG_DIR/*.json; do
	substring=true;
	for SUB in "$@"; do
		if [[ "$STR" != *"$SUB"* ]]; then
			substring=false;
		fi;
	done;
	if [ "$substring" == true ]; then
		echo "Training with config $STR";
		python train.py --config $STR;
		if [ $? -eq 0 ]; then
			echo "Testing with config $STR";
			python -Wignore inference.py --config $STR;
			if [ $? -eq 0 ]; then
				mv -v $STR $CONFIG_DIR/al_run;
			fi;
		fi;
	fi;
done
