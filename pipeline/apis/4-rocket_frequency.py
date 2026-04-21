#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket.
"""

import requests


def get_json(url):
    """Helper to fetch JSON data."""
    return requests.get(url, verify=False).json()


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"

    launches = get_json(launches_url)

    # Count launches per rocket ID
    rocket_count = {}

    for launch in launches:
        rocket_id = launch.get("rocket")
        rocket_count[rocket_id] = rocket_count.get(rocket_id, 0) + 1

    # Convert rocket IDs → names
    rocket_names = {}

    for rocket_id in rocket_count:
        rocket_data = get_json(rockets_url + rocket_id)
        rocket_names[rocket_id] = rocket_data.get("name")

    # Build final list (name, count)
    result = []

    for rocket_id, count in rocket_count.items():
        result.append((rocket_names[rocket_id], count))

    # Sort: by count DESC, then name ASC
    result.sort(key=lambda x: (-x[1], x[0]))

    # Print output
    for name, count in result:
        print("{}: {}".format(name, count))
