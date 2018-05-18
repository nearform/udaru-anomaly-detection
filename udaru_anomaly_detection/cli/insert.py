
import datetime

from udaru_anomaly_detection.trail.insert import trail_insert
from udaru_anomaly_detection.tests.generator import generate_resource


def insert(args):
    for resource_i, resource in enumerate(generate_resource(100, 'train')):
        print(f'insert [train]: {resource}')

        trail_insert(
            when=(datetime.datetime(2017, 1, 1) +
                  datetime.timedelta(days=1) * resource_i),
            who={
                'id': 'resource_user',
                'ip-address': '64.64.117.58'
            },
            what='action',
            subject=resource,
            meta={
                'dataset': 'train',
                'expect': 'NA'
            }
        )

    for resource_i, resource in enumerate(generate_resource(10, 'test')):
        print(f'insert [test]: {resource}')
        trail_insert(
            when=(datetime.datetime(2018, 1, 1) +
                  datetime.timedelta(days=1) * resource_i),
            who={
                'id': 'resource_user',
                'ip-address': '64.64.117.58'
            },
            what='action',
            subject=resource,
            meta={
                'dataset': 'test',
                'expect': 'valid'
            }
        )

    invalid_resources = [
     '../../../passwd',
     ':(){ :|: & };:',
     'a',
     'a' * 70,
     'res::ricky:/sl/jennifersaunders',
     'res:/sl/:ricky:/jennifersaunders'
    ]

    for resource_i, resource in enumerate(invalid_resources):
        print(f'insert [test]: {resource}')
        trail_insert(
            when=(datetime.datetime(2018, 2, 1) +
                  datetime.timedelta(days=1) * resource_i),
            who={
                'id': 'resource_user',
                'ip-address': '64.64.117.58'
            },
            what='action',
            subject=resource,
            meta={
                'dataset': 'test',
                'expect': 'invalid'
            }
        )

    nyc_ipaddress = '64.64.117.58'  # New York City
    wdc_ipaddress = '173.239.197.169'  # Washington DC
    lon_ipaddress = '5.101.142.229'  # London

    ip_inserts = [
        (nyc_ipaddress, lon_ipaddress, 9, True),
        (nyc_ipaddress, wdc_ipaddress, 2, True),
        (nyc_ipaddress, lon_ipaddress, 2, False)
    ]

    for user_i, (from_ip, to_ip, duration, valid) in enumerate(ip_inserts):
        print(f'insert [test]: {from_ip} -> {to_ip}: {duration}h')
        trail_insert(
            when=(datetime.datetime(2018, 3, 1) +
                  datetime.timedelta(days=1) * user_i),
            who={
                'id': f'user_{user_i}',
                'ip-address': from_ip
            },
            what='action',
            subject='res:bb185024/iptest',
            meta={
                'dataset': 'test',
                'expect': 'valid'
            }
        )

        trail_insert(
            when=(datetime.datetime(2018, 3, 1) +
                  datetime.timedelta(days=1) * user_i +
                  datetime.timedelta(hours=duration)),
            who={
                'id': f'user_{user_i}',
                'ip-address': to_ip
            },
            what='action',
            subject='res:bb185024/iptest',
            meta={
                'dataset': 'test',
                'expect': 'valid' if valid else 'invalid'
            }
        )
