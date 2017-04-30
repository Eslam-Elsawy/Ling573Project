#!/bin/bash

python3 src/content_selection/pagerank.py
python3 src/reranker/reranker.py

./evaluation/ROUGE/ROUGE-1.5.5.pl -e evaluation/ROUGE/data -a -n 4 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d evaluation/config/D3/devtest/config.xml > evaluation/output/D3/devtest/output.txt
