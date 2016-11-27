#!/bin/bash
dir=$1
desh=${dir: -1}
if [ "$desh" != "/" ]; then
	dir="$1/"
fi
python2.7 word2vec_unsupervised.py --mode "test" --title_doc $dir"title_StackOverflow.txt" --pred_pairs $dir"check_index.csv" \
--content $dir"docs.txt" --pred_out $2