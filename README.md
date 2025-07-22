# AI-Powered-Real-Time-Parathyroid-Gland-Detection-Tracking-During-ICG-Assisted-Thyroidectomy
Lightweight ensemble model combining OpenCV, ORB features, CSRT tracker, and Kalman filtering for real-time detection and robust tracking of parathyroid glands using fluorescence imaging during surgery.

# AI-Powered Real-Time Parathyroid Gland Detection & Tracking

This project implements a real-time parathyroid gland detection and tracking system designed to assist surgeons during ICG-assisted thyroidectomy procedures. By combining fluorescence imaging with computer vision techniques like ORB feature matching, CSRT tracking, and Kalman filtering, this ensemble approach achieves high accuracy under variable lighting conditions (NIR to normal light).

## 🔬 Background

- Parathyroid gland preservation during thyroidectomy is critical.
- Under NIR light, PGs fluoresce green when ICG is administered.
- Manual identification is error-prone due to occlusions and anatomical variations.
- Our model provides AI-assisted visual support without interfering with surgical judgment.

## ⚙️ Methodology

1. **Preprocessing**:
   - Convert frames to HSV color space
   - Apply Gaussian blur and ORB keypoint extraction

2. **Detection (NIR mode)**:
   - Fluorescence-based green masking
   - Morphological processing
   - Contour detection

3. **Tracking**:
   - Initial tracking via CSRT
   - Augmented with Kalman Filter for prediction across occlusions
   - ORB matching confirms lost tracks post lighting change

## 📈 Results

| Metric     | CSRT Only | Ensemble Model |
|------------|-----------|----------------|
| Accuracy   | 77.57%    | **96.78%**     |
| Precision  | 92.83%    | **95.18%**     |
| Recall     | 65.19%    | **99.32%**     |
| F1-Score   | 76.59%    | **97.20%**     |
| Processing | -         | ~40ms/frame    |

## 📦 Tech Stack

- Python
- OpenCV
- Kalman Filtering
- ORB Feature Matching
- CSRT Tracker

## 🩺 Use Case

> Supports surgeons in accurate gland identification during surgery, minimizing the risk of postoperative complications like hypoparathyroidism.




