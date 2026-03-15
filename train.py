from ultralytics import YOLO

if __name__ == '__main__':
  # Load a model
  #model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
  model = YOLO("./Pre-trained_Models/Seg/yolov8n-seg.pt")  # load a pretrained model (recommended for training)
  #model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

  # Train the model
  results = model.train(data="momo640.yaml", epochs=100, imgsz=640)