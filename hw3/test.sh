#!/bin/bash
dir=$1
desh=${dir: -1}
if [ "$desh" != "/" ]; then
	dir="$1/"
fi
python ./cnn.py --label_dat "$1all_label.p" --unlabel_dat "$1all_unlabel.p" --test_dat "$1test.p" --model $2 --mtype "test" --output $3