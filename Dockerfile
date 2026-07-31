FROM python:3.14-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/atcroster

RUN addgroup --system atcroster && adduser --system --ingroup atcroster atcroster
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .
RUN chown -R atcroster:atcroster /srv/atcroster
USER atcroster

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=3)"

CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "--threads=8", "wsgi:application"]
