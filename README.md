# Cell Segmentation and Membrane Intensity Measurement
## Usage: 
1. Go to "cellpose_tools" folder and run "pip install -e ."
2. Command line to run this script: cellpose-tools -p "your single channel picture" -o "output name" -j "job name" (--no-log if you don't need log file) --diameter 50 --band-dilation 1 --band-erotion 1