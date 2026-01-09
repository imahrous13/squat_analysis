# Cloudflare Tunnel URL

**Current Public URL:** https://ly-their.trycloudflare.com

**Created:** 2026-01-07 07:34 AM

**Status:** Active ✅

---

## How to Use:
1. Make sure Streamlit app is running: `streamlit run streamlit_app.py --server.port 8503`
2. Make sure tunnel is running: `cloudflared tunnel --url http://localhost:8503`
3. Share the URL above with anyone to access your Squat Analyzer app!

## Note:
- This URL is temporary and will change each time you restart the tunnel
- Both the Streamlit app AND the tunnel must be running for the URL to work
- If you get a 502 error, make sure the Streamlit app is running on port 8503

## Recent Changes:
- ✅ Removed black box overlay - now transparent!
