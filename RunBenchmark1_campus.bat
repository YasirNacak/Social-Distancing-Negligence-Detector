@echo off
cd social_distancing_negligence_detector
python DetectorBenchmark.py -input=Inputs/campus_test_footage.mp4 -model=yolov4 -video=false
PAUSE