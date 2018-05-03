
# Anomaly Detection

## Resource anomaly detection

Based on work in [_Anomaly Detection of Web-based Attacks_](https://www.cs.ucsb.edu/~vigna/publications/2003_kruegel_vigna_ccs03.pdf).

### Length Validation

Checks if the length of a `resource` string matches previuse observations.

```python
from anomaly_detection import check_length

length_model = check_length.train(dataset)
check_length.validate(
  length_model, 'res:fda1cf88:steven:/ak'
)
```

### Charecter Distribution Validation

Checks if the charecter combinations of a `resource` string matches previuse
observations.

```python
from anomaly_detection import check_distribution

distribution_model = check_distribution.train(dataset)
distribution_model.validate(
  distribution_model, 'res:fda1cf88:steven:/ak'
)
```

### Gramma Validation

Builds a graph model (think, regular expression) based on previuse
observations and checks if the new `resource` string maches·

TODO:

* Training the model can be solve. (1 hour for 100 resources).

```python
from anomaly_detection import check_gramma

distribution_model = check_gramma.train(dataset)
check_gramma.validate(
  distribution_model, 'res:fda1cf88:steven:/ak'
)
```

## IP-address Validation

Calculates the distance between the geolocations for the IP-addresses.
Based on that distance and the timestamps, it calculates the required velocity
and matches that against the speed of sound.

TODO:

* Automatially update the GeoLite2-City database weekly
* Be aware of licenses issue regarding GeoLite2-City.

```python
from anomaly_detection import check_ipaddress

hour = 60 * 60 * 1000
nyc_ipaddress = '64.64.117.58'  # New York City
lon_ipaddress = '5.101.142.229'  # London

print(f'{nyc_ipaddress} (New York City) -> {lon_ipaddress} (London) : 9 hours')
print(check_ipaddress.validate(previuse_timestamp_ms=0 * hour,
                               previuse_ipaddress=nyc_ipaddress,
                               current_timestamp_ms=9 * hour,
                               current_ipaddress=lon_ipaddress)) # Valid

print(f'{nyc_ipaddress} (New York City) -> {lon_ipaddress} (London) : 2 hours')
print(check_ipaddress.validate(previuse_timestamp_ms=0 * hour,
                               previuse_ipaddress=nyc_ipaddress,
                               current_timestamp_ms=2 * hour,
                               current_ipaddress=lon_ipaddress)) # Invalid
```

## LICENSE

* Private, Copyright (c) 2017 nearForm

* This product includes GeoLite2 data created by MaxMind, available from
 <a href="http://www.maxmind.com">http://www.maxmind.com</a>.
