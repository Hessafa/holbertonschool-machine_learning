#!/usr/bin/env python3
"""
Script that prints the location of a GitHub user.
"""

import requests
import sys
import time


if __name__ == "__main__":
    url = sys.argv[1]

    response = requests.get(url)

    # User not found
    if response.status_code == 404:
        print("Not found")

    # Rate limit exceeded
    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        current_time = int(time.time())
        minutes = (reset_time - current_time) // 60

        print("Reset in {} min".format(minutes))

    # Success
    else:
        data = response.json()
        print(data.get("location"))
