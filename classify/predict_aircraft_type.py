import pickle
from math import sin, cos, acos, radians
import numpy as np
import pandas as pd


def lat_long_dist(lat1, lon1, lat2, lon2, in_radians=False):
    """Calculate distance in km between pairs of latitude and longitude coords

    Args:
        lat1 (float):
            Latitude of origin
        lon2 (float):
            Longitude of origin
        lat2 (float):
            Latitude of destination
        lon2 (float):
            Longitude of origin
        in_radians (bool, optional):
            Set in_radians=True if coordinates are expressed in radians

    Returns:
        float: Distance in kilometers between origin and destination
    """
    # approximate radius of earth
    R = 6371.0

    if not in_radians:
        # convert from degrees to radians
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

    return R * acos(
        sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))


def load_airport_coords(path='airports.csv'):
    """Loads dataset of airport coordinates

    Args:
        path (str, optional): Path to csv containing airport coordinates

    Returns:
        Dictionary mapping airport coordinate string (3 characters) to array
        with format [lat, long]
    """
    vals = pd.read_csv(path, header=None).values
    coords = {vals[i][0]: vals[i][1:] for i in range(vals.shape[0])}
    return coords


def predict_aircraft_type(
        model_path, dest_airport_code, n_days_with_flights,
        aircraft_identifier_map, coords_path='airports.csv'):
    """
    Uses model to predict aircraft to serve route between YYZ and destination
    city

    Args:
        model_path (str):
            Path to classification model
        dest_airport_code (str):
            Three letter airport code
        n_days_with_flights (int):
            Number of days per week with 1 or more flights
        aircraft_identifier_map (dict):
            Dictionary mapping aircraft model name to numerical class used in
            classification model
        coords_path (str, optional):
            Path to csv containing airport coordinates

    Returns:
        List[str]:
            List of aircraft models that would be suitable to serve the route
    """
    # load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # calculate features
    coords = load_airport_coords(coords_path)
    lat_YYZ, lon_YYZ = coords['YYZ']
    lat_dest, lon_dest = coords[dest_airport_code]

    dist = lat_long_dist(lat_YYZ, lon_YYZ, lat_dest, lon_dest)
    features = np.array([[dist, n_days_with_flights, lat_dest, lon_dest]])

    # predict class
    aircraft_type = model.predict(features)
    aircraft_models = [
        x for x, y in aircraft_identifier_map.items() if y == aircraft_type]

    return aircraft_models
