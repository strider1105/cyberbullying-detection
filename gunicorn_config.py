import os

# Bind to PORT if provided, otherwise default to 5000
port = int(os.environ.get("PORT", 5000))
bind = f"0.0.0.0:{port}"

# Number of worker processes
workers = 4

# Number of threads per worker
threads = 2

# Timeout in seconds
timeout = 120

# Logging
accesslog = "-"
errorlog = "-"

# Reload workers when code changes (development only)
reload = os.environ.get("ENVIRONMENT") == "development"