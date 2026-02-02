FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests pandas

COPY frontend/ui.py /app/ui.py

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
