#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch with details.
"""

import requests


def get_json(url):
    """Helper to get JSON data with SSL disabled."""
    return requests.get(url, verify=False).json()


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads/"

    launches = get_json(launches_url)

    # Sort launches by date_unix
    launches.sort(key=lambda x: x.get("date_unix", float("inf")))

    first = launches[0]

    # Fetch rocket info
    rocket = get_json(rockets_url + first.get("rocket"))
    rocket_name = rocket.get("name")

    # Fetch launchpad info
    launchpad = get_json(launchpads_url + first.get("launchpad"))
    launchpad_name = launchpad.get("name")
    launchpad_locality = launchpad.get("locality")

    # Output
    print("{} ({}) {} - {} ({})".format(
        first.get("name"),
        first.get("date_local"),
        rocket_name,
        launchpad_name,
        launchpad_locality
    ))
