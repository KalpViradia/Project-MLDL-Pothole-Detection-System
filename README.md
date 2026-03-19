# 🕳️ Pothole Detection System

> AI-powered road damage detection using **YOLOv8** and **FastAPI**, with a modern **Next.js** frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?logo=fastapi&logoColor=white)

---

## 📖 Overview

This system detects potholes in road images and videos using a custom-trained **YOLOv8 large (v8l)** model. It provides:

- **Image detection** — Upload an image or provide a URL, and get annotated results with bounding boxes and confidence scores.
- **Video detection** — Upload a video and receive frame-by-frame analysis with pothole annotations.
- **Reporting API** — Submit detected pothole locations with coordinates for mapping.

---

## 📁 Project Structure

```
Pothole Detection System/
├── backend/                    # FastAPI backend server
│   ├── main.py                 # API endpoints (detect, detect-video, report)
│   ├── models/                 # YOLOv8 model weights (not tracked in Git)
│   ├── notebooks/              # Training notebooks (Colab/Kaggle)
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # Next.js 16 web application
│   ├── app/
│   │   ├── page.tsx            # Landing page
│   │   ├── image/page.tsx      # Image detection page
│   │   ├── video/page.tsx      # Video detection page
│   │   ├── about/page.tsx      # About page
│   │   ├── journey/page.tsx    # Project journey page
│   │   ├── components/         # Shared components (Navbar)
│   │   ├── globals.css         # Global styles & design system
│   │   └── layout.tsx          # Root layout
│   └── package.json
│
├── dataset_construction/       # Dataset preparation pipeline
│   ├── scripts/                # Data processing & conversion scripts
│   │   ├── build_hybrid_dataset.py
│   │   ├── convert_csv_to_yolo.py
│   │   ├── convert_rdd2022_to_yolo.py
│   │   ├── convert_xml_to_yolo.py
│   │   ├── dataset_splitter.py
│   │   ├── merge_datasets.py
│   │   ├── prepare_dataset_768.py
│   │   ├── remove_duplicates.py
│   │   ├── resize_and_standardize.py
│   │   └── validate_dataset.py
│   └── Docs/                   # Dataset documentation
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **CUDA** (optional, for GPU acceleration)

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Place your trained model at backend/models/best.pt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000` and connects to the backend at `http://localhost:5000`.

---

## 🔌 API Endpoints

| Method | Endpoint             | Description                        |
| ------ | -------------------- | ---------------------------------- |
| GET    | `/api/health`        | Health check & device info         |
| POST   | `/api/detect`        | Detect potholes in a single image  |
| POST   | `/api/detect-video`  | Detect potholes in a video         |
| POST   | `/api/report`        | Submit a pothole report            |
| GET    | `/api/reports`       | Retrieve all submitted reports     |

### Image Detection

```bash
# Via file upload
curl -X POST http://localhost:5000/api/detect -F "file=@road_image.jpg"

# Via image URL
curl -X POST http://localhost:5000/api/detect -F "image_url=https://example.com/road.jpg"
```

### Video Detection

```bash
curl -X POST http://localhost:5000/api/detect-video \
  -F "file=@road_video.mp4" \
  -F "sample_rate=15"
```

---

## 🧠 Model Details

| Property       | Value                  |
| -------------- | ---------------------- |
| Architecture   | YOLOv8 Large (v8l)     |
| Input Size     | 768 × 768              |
| Dataset        | ~19,000 images         |
| Classes        | Pothole (single-class) |
| Conf Threshold | 0.35                   |
| IoU Threshold  | 0.45                   |

---

## 📊 Dataset Construction

The `dataset_construction/` folder contains the complete pipeline used to build the training dataset:

1. **Format conversion** — CSV, XML, and RDD2022 annotations → YOLO format
2. **Merging** — Combine multiple source datasets
3. **Deduplication** — Remove duplicate images
4. **Resizing** — Standardize to 768×768
5. **Splitting** — Train/Val/Test split
6. **Validation** — Verify dataset integrity

---

## 🛠️ Tech Stack

- **Detection Model**: YOLOv8 (Ultralytics)
- **Backend**: FastAPI, OpenCV, PyTorch
- **Frontend**: Next.js 16, React 19, TypeScript
- **Styling**: CSS with custom design system

---

## 📝 License

This project is for academic purposes — **Machine Learning and Deep Learning Fundamentals and Applications** course project.
