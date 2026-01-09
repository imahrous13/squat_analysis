@echo off
echo Starting Cloudflare Tunnel for Streamlit App...
echo.
echo Your Streamlit app will be accessible via a public Cloudflare URL
echo Look for the https://....trycloudflare.com URL below:
echo ================================================================
echo.

cloudflared tunnel --url http://localhost:8503