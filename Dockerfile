FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV GEMINI_API_KEY=${GEMINI_API_KEY}

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

