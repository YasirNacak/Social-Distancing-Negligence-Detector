@echo off
cd social_distancing_negligence_detector
python DetectorBenchmark.py -input=Inputs/church_street_test_footage.mp4 -model=yolov4 -video=true
PAUSE