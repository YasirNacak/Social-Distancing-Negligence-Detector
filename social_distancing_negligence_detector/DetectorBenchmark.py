import time
import sys

import cv2

from Detector import Detector, Models


def print_usage_and_exit():
    print("Invalid Arguments!")
    print("Usage: python DetectorBenchmark.py -input=[path-to-video] -model=[model-name] -video=[true/false]")
    print()
    print("Available Models:")
    for model in Models().get_models():
        print(model)
    exit(-1)


def check_args():
    input_file_name = ""
    model_name = ""
    show_video = True

    for arg in sys.argv[1:]:
        if arg.startswith("-input"):
            input_file_name = arg[arg.rfind("=") + 1 :]
        elif arg.startswith("-model"):
            model_name = arg[arg.rfind("=") + 1 :]
        elif arg.startswith("-video"):
            arg_value = arg[arg.rfind("=") + 1 :]
            if arg_value == "true":
                show_video = True
            elif arg_value == "false":
                show_video = False
            else:
                print_usage_and_exit()
        else:
            print_usage_and_exit()

    if input_file_name == "" or model_name == "":
        print_usage_and_exit()

    return input_file_name, model_name, show_video


def main():
    input_file_name, model_name, show_video = check_args()

    print(f"Input Video: {input_file_name}")
    print(f"Model Name: {model_name}")

    detector = Detector()
    detector.initialize(model_name, input_file_name)

    can_read_frames = True
    current_frame = 0

    start_time = time.time()
    while can_read_frames:
        current_frame += 1
        can_read_frames, frame = detector.analyze_frame(current_frame)

        print(f"\rAnalyzing Frame {current_frame}", end='')
        if current_frame % 3 == 0:
            print(" \\", end='')
        elif current_frame % 3 == 1:
            print(" |", end='')
        elif current_frame % 3 == 2:
            print(" /", end='')
        if show_video and frame is not None:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

    analyze_time = int(time.time() - start_time)

    print()
    print(f"Total Frames: {current_frame}")
    print(f"Analyze Time: {analyze_time}s")
    print(f"Average Frames Per Second: {current_frame // analyze_time}")


if __name__ == '__main__':
    main()
