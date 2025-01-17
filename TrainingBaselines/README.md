# Steps to reproduce training results

1. Install yolov5 with: `pip install yolov5`
2. Find the installation (if using venv, it will be in the `site-packages` folder. Modify the file material_data.yaml accordingly
3. Run the script generate_YOLO_dataset.py for version material_version and for both modes (train & val)
4. Copy train & val images to the directories specified in the .yaml file
5. Train with: `yolov5 train --data material_data.yaml --img 480 --rect --batch -1 --weights turhancan97/yolov5-detect-trash-classification --epochs 300 --save-period 10 --project trashcan_from_pretrained --seed 17`

   Notes on arguments: `--batch -1` will automatically determine the optimal batch size based on your hardware. `--seed 17` is set for reproducibility.
   The weights are taken from: https://huggingface.co/turhancan97/yolov5-detect-trash-classification
   Documentation of YOLOv5: https://docs.ultralytics.com/yolov5/