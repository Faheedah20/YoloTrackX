# 🎯 YoloTrackX

**Multi-class Object Detection & Tracking with YOLOv8**

A clean, modern Streamlit app for real-time object detection and multi-object tracking.  
Upload images or videos, use your webcam, filter classes, and export results — all with a beautiful **burgundy · gold · cream** UI.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **YOLOv8 models** | Nano → XLarge (auto-download) |
| 🔍 **Tracking** | ByteTrack & BoT-SORT with persistent IDs |
| 🖼️ **Inputs** | Image, Video, Webcam |
| 🎛️ **Controls** | Confidence, IoU, class filter, display toggles |
| 📊 **Analytics** | Live FPS, object counts, unique tracks, Plotly charts |
| ⬇️ **Export** | Annotated image/video + CSV of detections |
| 🎨 **UI** | Bright burgundy / gold / cream theme |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YoloTrackX.git
cd YoloTrackX

# 2. Virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py