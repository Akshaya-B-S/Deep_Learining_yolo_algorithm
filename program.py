from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in the video
results = model.predict(
    source="input.mp4",
    conf=0.5,
    save=True,
    show=True
)

print("Detection completed!")