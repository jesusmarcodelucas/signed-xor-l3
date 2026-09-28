#!/usr/bin/env bash
# Download MNIST (IDX, gzip) into data/
set -e
mkdir -p data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "data/$f.gz" ] || curl -L -o "data/$f.gz" "https://raw.githubusercontent.com/fgnt/mnist/master/$f.gz"
done
echo "MNIST ready in data/"
