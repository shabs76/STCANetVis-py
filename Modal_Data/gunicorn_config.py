# gunicorn_config.py

workers = 4  # Number of worker processes
bind = "0.0.0.0:8000"  # IP address and port to bind to
timeout = 120  # Maximum time a request is allowed to process