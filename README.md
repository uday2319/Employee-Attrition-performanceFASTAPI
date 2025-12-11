Employee Attrition Project — FastAPI (Deployed on Render)

Live: Deployed on Render — https://your-app-name.onrender.com (replace with your actual Render URL)

A small FastAPI service for the Employee Attrition demo, deployed to Render. 

Features:

 *FastAPI backend exposing REST endpoints (/predict).
*Example model loading flow (placeholder for your model.pkl / columns.pkl).
*Ready for production with Gunicorn + Uvicorn workers.

Tech stack :

1.Python 3.9+ (or your chosen version)
2.FastAPI
3.Uvicorn (for local dev)
4.Gunicorn + uvicorn.workers.UvicornWorker (for production on Render)
5.scikit-learn, pandas, numpy, etc...
