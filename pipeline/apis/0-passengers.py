#!/usr/bin/env python3
"""
Module that provides a function to retrieve starships
that can carry a given number of passengers using the SWAPI API.
"""

import requests


def availableShips(passengerCount):
    """
    Returns a list of starship names that can hold at least
    passengerCount passengers.

    Args:
        passengerCount (int): minimum number of passengers

    Returns:
        list: list of ship names
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")

            # Skip invalid values
            if passengers.lower() in ["unknown", "n/a", "none"]:
                continue

            # Remove commas and convert
            try:
                passengers = int(passengers.replace(",", ""))
            except ValueError:
                continue

            if passengers >= passengerCount:
                ships.append(ship.get("name"))

        # Pagination
        url = data.get("next")

    return ships
