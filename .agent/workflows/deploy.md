---
description: How to deploy the Gym Analyzer project to Vercel or Streamlit Cloud
---

# Deployment Guide

This project can be deployed to several platforms. **Streamlit Cloud** is highly recommended due to the size of the dependencies (MediaPipe and YOLO).

## Option 1: Streamlit Cloud (Recommended)
This is the easiest way to deploy.

1. Push your code to your GitHub repository (already done!).
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select your `squat_analysis` repository and `streamlit_app.py` as the main file.
5. Click **Deploy**.

## Option 2: Vercel
Deploying Streamlit to Vercel requires some configuration.

### 1. Update `requirements.txt`
Vercel's Linux environment requires the "headless" version of OpenCV. 
- Change `opencv-python` to `opencv-python-headless`.

### 2. Create `vercel.json`
Create a file named `vercel.json` in the root directory:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "streamlit_app.py",
      "use": "@vercel/python",
      "config": { "maxLambdaSize": "250mb", "runtime": "python3.9" }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "streamlit_app.py"
    }
  ]
}
```

### 3. Build Command Note
Vercel isn't natively designed to run Streamlit's long-running server. You may need to use a template like `streamlit-vercel` which wraps the app in a Flask or FastAPI container. 

**Warning:** This project uses `ultralytics` (YOLO) and `mediapipe`. These packages together exceed Vercel's 250MB limit. You will likely see a "deployment too large" error on Vercel unless you use an external API for the processing.

## Recommendation
For a computer vision app of this size, **Docker-based hosting** (like Google Cloud Run, Railway, or Hugging Face Spaces) or **Streamlit Cloud** are the most stable options.
