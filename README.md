
# License Plate Detection using TensorFlow Object Detection API

This repository demonstrates the implementation of a custom license plate detection model using the TensorFlow Object Detection API. The model is trained on a custom dataset and exported to TensorFlow Lite for efficient inference. It includes steps for dataset preparation, model training, and quantization for deployment on edge devices.

---

## Features

- Fine-tunes `ssd_mobilenet_v2_320x320_coco17_tpu-8` for license plate detection.
- Custom dataset integration with `Roboflow` API.
- Generates TFRecord files from labeled images.
- Supports quantization for TensorFlow Lite (TFLite) and NPU acceleration.

---

## Requirements

- Python 3.x
- TensorFlow 2.8.0
- Roboflow API Key
- Required Python packages (see `requirements.txt`).

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Steps to Run

### 1. Clone the repository
```bash
git clone https://github.com/1adarshkv9/licenseplatedetection.git
cd licenseplatedetection
```

### 2. Dataset Preparation
- Use the [Roboflow](https://roboflow.com/) API to download and preprocess the dataset.

### 3. Training
- Modify the `pipeline_file.config` for your dataset and training requirements.
- Train the model:
```bash
python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=path_to_pipeline_file.config \
    --model_dir=path_to_training_dir \
    --alsologtostderr \
    --num_train_steps=10000 \
    --sample_1_of_n_eval_examples=1
```

### 4. Export to TensorFlow Lite
- Export the trained model to TensorFlow Lite format:
```bash
python /content/models/research/object_detection/export_tflite_graph_tf2.py \
    --trained_checkpoint_dir=path_to_training_dir \
    --output_directory=path_to_output_dir \
    --pipeline_config_path=path_to_pipeline_file.config
```

- Perform quantization for NPU acceleration:
```python
converter = tf.lite.TFLiteConverter.from_saved_model('path_to_saved_model')
# Custom quantization code here...
quantized_tflite_model = converter.convert()
```

### 5. Visualize Training Metrics
- Launch TensorBoard:
```bash
tensorboard --logdir path_to_training_logs
```

---

## Outputs

- Trained model checkpoints.
- Quantized TFLite model: `quant_model_NPU.tflite`.

Below is an example detection output from the i.MX8MP device using the trained model:

![](https://github.com1adarshkv9/licenseplatedetection/main/Video2024-10-07at5.47.11PM-ezgif.com-crop.gif)

The model successfully detects and localizes license plates in real-world images with high accuracy.

---

## Acknowledgments

- TensorFlow Object Detection API ([models repository](https://github.com/tensorflow/models))
- [Roboflow](https://roboflow.com/)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or feedback, feel free to reach out at **adarsh.k@phytecembedded.in**.

---


