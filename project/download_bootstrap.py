# download_bootstrap.py
import urllib.request
import os

# Create directories
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

print("Downloading Bootstrap CSS...")
urllib.request.urlretrieve(
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'static/css/bootstrap.min.css'
)

print("Downloading Bootstrap JS...")
urllib.request.urlretrieve(
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
    'static/js/bootstrap.bundle.min.js'
)

print("✅ Bootstrap downloaded successfully!")
print("Files saved to:")
print("  - static/css/bootstrap.min.css")
print("  - static/js/bootstrap.bundle.min.js")