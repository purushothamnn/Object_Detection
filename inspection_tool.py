import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QHBoxLayout, QStatusBar, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsTextItem
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QFont, QIcon
from PyQt6.QtCore import Qt, QRectF

# COCO Class Labels
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Load YOLOv8 Model
try:
    model = YOLO("yolov8s.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.detectionResults = []
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("AI-Powered Object Detection Tool")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #2c3e50; color: white; font-family: Arial;")
        
        # Sidebar
        self.sidebar = QFrame(self)
        self.sidebar.setFixedWidth(270)
        self.sidebar.setStyleSheet("background-color: #34495e; padding: 10px;")
        self.sidebarLayout = QVBoxLayout(self.sidebar)

        self.uploadButton = QPushButton("📁 Upload Image")
        self.uploadButton.setStyleSheet("background-color: #1abc9c; color: white; padding: 10px;")
        self.uploadButton.clicked.connect(self.loadImage)

        self.detectButton = QPushButton("🚀 Detect Objects")
        self.detectButton.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px;")
        self.detectButton.setEnabled(False)
        self.detectButton.clicked.connect(self.detectObjects)

        self.statusBar = QStatusBar(self)
        self.statusBar.setStyleSheet("background-color: #34495e; color: white;")
        self.statusBar.showMessage("Ready")

        self.sidebarLayout.addWidget(self.uploadButton)
        self.sidebarLayout.addWidget(self.detectButton)
        self.sidebarLayout.addStretch()
        self.sidebarLayout.addWidget(self.statusBar)

        # Main Area (Image Display)
        self.graphicsView = QGraphicsView()
        self.graphicsView.setStyleSheet("background-color: #2c3e50; border: 2px solid #34495e;")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.pixmapItem = None

        # Layout
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.sidebar)
        self.layout.addWidget(self.graphicsView)

        self.setLayout(self.layout)

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        
        if filePath:
            self.image = cv2.imread(filePath)
            self.displayImage(self.image)
            self.detectButton.setEnabled(True)
            self.statusBar.showMessage("Image Loaded Successfully")

    def detectObjects(self):
        if self.image is None:
            return
        
        self.statusBar.showMessage("Detecting Objects...")

        results = model(self.image)  # Run YOLOv8 detection
        self.detectionResults = results[0].boxes.data.cpu().numpy()  # Extract bounding boxes correctly

        self.displayImage(self.image, draw_bboxes=True)
        self.statusBar.showMessage("Detection Complete")

    def displayImage(self, img, draw_bboxes=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if self.pixmapItem:
            self.scene.clear()

        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmapItem)

        if draw_bboxes:
            self.drawBoundingBoxes()

        self.graphicsView.fitInView(self.pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)

    def drawBoundingBoxes(self):
        for box in self.detectionResults:
            x1, y1, x2, y2, conf, class_id = box
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            class_id = int(class_id)
            object_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Unknown {class_id}"

            rectItem = InteractiveBoundingBox(rect, object_name, conf, self.scene)  # Pass scene instead of parent
            self.scene.addItem(rectItem)

class InteractiveBoundingBox(QGraphicsRectItem):
    def __init__(self, rect, object_name, confidence, scene):
        super().__init__(rect)
        self.object_name = object_name
        self.confidence = confidence
        self.setPen(QPen(QColor(255, 0, 0), 3))
        self.setAcceptHoverEvents(True)

        self.textItem = QGraphicsTextItem(f"{self.object_name} ({self.confidence:.2f})")
        self.textItem.setDefaultTextColor(Qt.GlobalColor.white)
        self.textItem.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.textItem.setPos(rect.x() + 5, rect.y() - 22)
        self.textItem.hide()

        scene.addItem(self.textItem)  # Correctly add to scene

    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor(0, 255, 0), 3))  # Green highlight on hover
        self.textItem.show()

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(QColor(255, 0, 0), 3))  # Reset color
        self.textItem.hide()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())
