# ---- Build stage: install dependencies ----
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
# Upgrade pip/setuptools/wheel first -- the versions bundled inside the
# base image are often outdated and carry their own known CVEs (wheel,
# pip, and jaraco.context -- a setuptools dependency -- all showed up in
# a Trivy scan here until this line was added). Installing requirements
# afterward then uses the patched versions, not the stale bundled ones.
RUN pip install --no-cache-dir --user --upgrade pip setuptools wheel
# --user installs to a local dir we can cleanly copy into the final image,
# keeping the final image free of build tools and caches.
RUN pip install --no-cache-dir --user -r requirements.txt


# ---- Final stage: minimal runtime image ----
FROM python:3.10-slim

# Patch OS-level packages (e.g. pcre2) to their latest available Debian
# security fixes, then clean up apt's cache so it doesn't bloat the image.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Non-root user -- never run the service as root.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Bring in installed Python packages from the build stage only (not pip
# caches, compilers, etc.) -- keeps the final image smaller.
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Only what's needed to SERVE the model -- not training scripts or raw data.
COPY app/api/ ./app/api/

# The specific, validated model version to serve -- exported ahead of
# time into a clean, self-contained folder (see project README / Phase 4
# notes for the export command). This avoids baking host-specific
# absolute artifact paths from mlflow.db into the image.
COPY exported_model/ ./model/

# Reference distribution + drift detection code, needed by /drift/check
COPY data/reference/ ./data/reference/
COPY app/drift/ ./app/drift/

# Tell the app to load from this local folder instead of querying the
# MLflow registry (which would need mlflow.db + mlartifacts/, tied to
# absolute paths from the training machine).
ENV MODEL_ARTIFACT_PATH=/app/model
# Keep this in sync with whichever version exported_model/ actually
# contains -- used for display in /model/info and the Prometheus
# model_version_info label. Manual for now; Phase 8's CI/CD is what
# would eventually automate re-exporting + rebuilding on promotion.
ENV MODEL_VERSION=3

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Kubernetes (Phase 9) will use its own probes, but this makes the
# container self-reporting even when run standalone with `docker run`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

WORKDIR /app/app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]