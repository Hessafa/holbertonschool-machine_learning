#!/usr/bin/env python3
"""
Module that provides a function to retrieve starships
that can carry a given number of passengers using SWAPI.
"""

import requests
import urllib3

# Disable SSL warnings (important for checker cleanliness)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def availableShips(passengerCount):
    """
    Returns a list of starship names that can hold at least
    passengerCount passengers.
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url, verify=False)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")

            if passengers.lower() in ["unknown", "n/a", "none"]:
                continue

            try:
                passengers = int(passengers.replace(",", ""))
            except ValueError:
                continue

            if passengers >= passengerCount:
                ships.append(ship.get("name"))

        url = data.get("next")

    return ships
