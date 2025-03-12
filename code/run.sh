#!/bin/bash

python main1.py

# FFT and save the results.
cp ~/Documents/CMU/Research/EG_comparison/txtfiles/pkgg_test.txt ~/Documents/CMU/Software/FFTLog-master-slosar/test/
cd ~/Documents/CMU/Software/FFTLog-master-slosar/

make test_gg_EG.out
make clean

cd ~/Dropbox/CMU/Research/EG_comparison/code/

python main2.py
