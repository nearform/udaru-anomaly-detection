
import math
import os
import urllib.request
import os.path as path
import tarfile
import typing

from tqdm import tqdm
import geoip2.database

number_type = typing.Union[int, str]
dirname = path.dirname(path.realpath(__file__))
download_dir = path.join(dirname, 'download')
geolitedb_file_mmdb = path.join(download_dir, 'GeoLite2-City.mmdb')
geolitedb_file_targz = path.join(download_dir, 'GeoLite2-City.tar.gz')


def download(*args, desc=None, **kwargs):
    last_b = [0]

    def _download_hook(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(*args,
                                   reporthook=_download_hook, data=None,
                                   **kwargs)


# Automatically download the GeoLite2-City database
# TODO: check License requirements
# TODO: automatically update database file every week
if not path.exists(geolitedb_file_mmdb):
    os.makedirs(download_dir, exist_ok=True)

    download(
        'http://geolite.maxmind.com/download/geoip'
        '/database/GeoLite2-City.tar.gz',
        geolitedb_file_targz,
        desc='GeoLite2-City.tar.gz'
    )

    with tarfile.open(geolitedb_file_targz, 'r|gz') as tar_file:
        for file_info in tar_file:
            if file_info.name.endswith('/GeoLite2-City.mmdb'):
                with tar_file.extractfile(file_info) as fp_read:
                    with open(geolitedb_file_mmdb, 'wb') as fp_write:
                        fp_write.write(fp_read.read())

    os.remove(geolitedb_file_targz)

# Load database
reader = geoip2.database.Reader(geolitedb_file_mmdb)


def geolocate(ipaddress: str) -> typing.Tuple[float, float]:
    response = reader.city(ipaddress)
    return (
        response.location.latitude / 180 * math.pi,
        response.location.longitude / 180 * math.pi
    )


def haversine(angle: float) -> float:
    return math.sin(angle / 2) ** 2


def haversine_inverse(h: float) -> float:
    return 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def validate(previuse_timestamp_ms: number_type, previuse_ipaddress: str,
             current_timestamp_ms: number_type, current_ipaddress: str,
             threshold: float=343.0) -> bool:
    # threshold is the speed of sound in m/s

    # Compute time difference in seconds
    time_diff_sec = float(current_timestamp_ms - previuse_timestamp_ms) / 1000

    # Compute distance traveled
    previuse_lat, previuse_lon = geolocate(previuse_ipaddress)
    current_lat, current_lon = geolocate(current_ipaddress)

    h = haversine(current_lat - previuse_lat) + \
        math.cos(previuse_lat) * math.cos(current_lat) * \
        haversine(current_lon - previuse_lon)

    earth_radius = 6.3781e6
    earth_distance = earth_radius * haversine_inverse(h)

    # Compute velocity
    velocity = earth_distance / time_diff_sec

    # velocity is negative in case of time travel
    return 0 <= velocity <= threshold
