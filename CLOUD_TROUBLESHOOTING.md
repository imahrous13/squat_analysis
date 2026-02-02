# Streamlit Cloud Troubleshooting Guide

## If you're seeing "Oh no" errors on Streamlit Cloud:

### 1. Check the Cloud Logs
- Go to your Streamlit Cloud dashboard
- Click on your app
- Click "Manage app" → "Logs"
- Look for the specific error message

### 2. Common Cloud Issues and Solutions

#### Issue: Memory Limit Exceeded
**Symptoms:** App crashes with "Oh no" during startup
**Solution:** The lazy loading we implemented should help, but you may need to:
- Disable the hybrid mode (YOLO) on Cloud
- Use only MediaPipe (lighter weight)

#### Issue: Missing System Dependencies
**Symptoms:** Import errors for cv2, mediapipe
**Solution:** Ensure `packages.txt` is in the root directory (✅ already there)

#### Issue: WebRTC Not Working on Cloud
**Symptoms:** Webcam tab doesn't work
**Solution:** WebRTC has limitations on Cloud. Consider:
- Using only the "Upload Video" tab on Cloud
- Or deploying to a different platform for webcam support

### 3. Recommended Cloud Settings

For Streamlit Cloud, edit your app settings:
- **Python version:** 3.11
- **Main file path:** `app.py`
- **Advanced settings:**
  - Keep default resource limits
  - The lazy loading should prevent memory issues

### 4. Test Locally First

Before deploying to Cloud, always test locally:
```bash
python -m streamlit run app.py
```

### 5. Disable Features for Cloud

If Cloud keeps crashing, you can disable heavy features:

**Option A:** Disable Hybrid Mode
- In the app sidebar, keep "Use AI-Enhanced Detection" unchecked
- This prevents YOLO from loading

**Option B:** Disable WebRTC (Webcam)
- Comment out the WebRTC tab in app.py
- Use only video upload mode

### 6. Alternative: Use Lighter Dependencies

If issues persist, consider:
- Using only MediaPipe (no YOLO)
- Reducing the number of analyzers loaded
- Processing videos at lower resolution

## Current Status

✅ **Lazy Loading Implemented**
- MediaPipe only loads when needed
- YOLO only loads when hybrid mode is enabled
- Analyzers load on-demand

✅ **Environment Variables Set**
- Protobuf optimized
- GPU disabled for Cloud compatibility
- OpenCV optimized

✅ **System Dependencies Configured**
- packages.txt includes all required libraries

## Need More Help?

If the app still doesn't work on Cloud, please share:
1. The exact error message from Cloud logs
2. At what point it crashes (startup, first use, etc.)
3. Which features you're trying to use
