import requests
import sys

def check(url, name):
    print(f"Checking {name}...")
    try:
        r = requests.get(url, timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Content snippet: {r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check("https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY", "APOD")
    check("https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol=1000&api_key=DEMO_KEY", "Mars Photos")
