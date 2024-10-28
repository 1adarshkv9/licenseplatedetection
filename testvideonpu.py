import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import time

def tflite_detect_video(modelpath, video_path, lblpath, min_conf=0.5):
    # Load labels
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load TFLite model with NPU delegate
    delegate_path = "/usr/lib/libvx_delegate.so"
    try:
        delegate = tflite.load_delegate(delegate_path)
        print("NPU delegate loaded successfully.")
    except Exception as e:
        print(f"Failed to load NPU delegate: {e}")
        return

    interpreter = tflite.Interpreter(model_path=modelpath, experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        start_time = time.time()  # Start timing
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform detection
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class indices
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence scores

        detections = []

        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            # Calculate elapsed time and FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        #print(f'FPS: {fps:.2f}')  # Print the current FPS

        # Display the result
        cv2.imshow('output', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
model_path = 'quant_model_NPU_3k.tflite'
video_path = 'demo12fps.webm'
label_path = 'labelmap.txt'
tflite_detect_video(model_path, video_path, label_path)

