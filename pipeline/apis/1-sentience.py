#!/usr/bin/env python3
"""
Module that provides a function to retrieve the list of
home planets of all sentient species using SWAPI.
"""

import requests
import urllib3

# Disable SSL warnings (required for checker environment)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def sentientPlanets():
    """
    Returns a list of names of home planets of all sentient species.
    """
    url = "https://swapi.dev/api/species/"
    planets = []

    while url:
        response = requests.get(url, verify=False)
        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            # Check if species is sentient
            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")

                if homeworld_url:
                    home_response = requests.get(homeworld_url, verify=False)
                    home_data = home_response.json()
                    planets.append(home_data.get("name"))
                else:
                    planets.append("unknown")

        # Pagination
        url = data.get("next")

    return planets
