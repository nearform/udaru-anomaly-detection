
from udaru_anomaly_detection import check_ipaddress

hour = 60 * 60 * 1000
nyc_ipaddress = '64.64.117.58'  # New York City
wdc_ipaddress = '173.239.197.169'  # Washington DC
lon_ipaddress = '5.101.142.229'  # London

print(f'{nyc_ipaddress} (New York City) -> {lon_ipaddress} (London) : 9 hours')
print(check_ipaddress.validate(previuse_timestamp_ms=0 * hour,
                               previuse_ipaddress=nyc_ipaddress,
                               current_timestamp_ms=9 * hour,
                               current_ipaddress=lon_ipaddress))

print('')
print(f'{nyc_ipaddress} (New York City) -> {wdc_ipaddress} (Washington DC) : 2 hours')
print(check_ipaddress.validate(previuse_timestamp_ms=0 * hour,
                               previuse_ipaddress=nyc_ipaddress,
                               current_timestamp_ms=2 * hour,
                               current_ipaddress=wdc_ipaddress))

print('')
print(f'{nyc_ipaddress} (New York City) -> {lon_ipaddress} (London) : 2 hours')
print(check_ipaddress.validate(previuse_timestamp_ms=0 * hour,
                               previuse_ipaddress=nyc_ipaddress,
                               current_timestamp_ms=2 * hour,
                               current_ipaddress=lon_ipaddress))
