#!/bin/bash

python3 src/run_on_patas/reranker.py
export JAVA_HOME=/opt/jdk8/bin/java
python3 src/run_on_patas/entitybasedreranker.py

cp outputs/entity_reranker_D4/devtest/* outputs/D4_devtest
cp outputs/entity_reranker_D4/eval/* outputs/D4_evaltest

./evaluation/ROUGE/ROUGE-1.5.5.pl -e evaluation/ROUGE/data -a -n 4 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d evaluation/config/D4/devtest/config.xml > evaluation/output/D4/devtest/output.txt

./evaluation/ROUGE/ROUGE-1.5.5.pl -e evaluation/ROUGE/data -a -n 4 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d evaluation/config/D4/eval_data/config.xml > evaluation/output/D4/eval_data/output.txt

cp evaluation/output/D4/eval_data/output.txt results/D4_evaltest_rouge_scores.out
cp evaluation/output/D4/devtest/output.txt results/D4_devtest_rouge_scores.out

