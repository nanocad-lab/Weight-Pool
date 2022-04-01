#!/bin/bash
pip install kiwisolver
git clone https://github.com/XJay18/QuickDraw-pytorch.git
mv QuickDraw-pytorch/DataUtils/prepare_data.py QuickDraw-pytorch
mv QuickDraw-pytorch/DataUtils/generate_data.py QuickDraw-pytorch
cd QuickDraw-pytorch
python3 prepare_data.py -c 100 -d 1
cd ..
echo "Quick Draw dataset has been downloaded and processed"
