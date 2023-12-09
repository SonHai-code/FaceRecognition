from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data='Dataset/SplitData/dataOffline.yaml')

if __name__ == "__main__":
    main()

