from ultralytics import YOLO

# load a model
model = YOLO('runs/detect/train23/weights/last.pt')

# Train the model
model.predict(data='yolo-heandping.yaml',source='dataset/images/test',workers=0,epochs=1000,batch=16)