#!/bin/bash

cd src/content_selection
python3 pagerank.py

cd ../reranker
python3 reranker.py