
import datetime
import os
import os.path as path
import pickle

from udaru_anomaly_detection import \
    check_length, \
    check_distribution as check_dist, \
    check_gramma, \
    check_ipaddress

from udaru_anomaly_detection.trail.read import trail_read

demo_ipaddress_helper = {
    '64.64.117.58': 'New York City',
    '173.239.197.169': 'Washington DC',
    '5.101.142.229': 'London'
}


def test(args):
    reader = trail_read(
        to_date=datetime.datetime.strptime(getattr(args, 'to'), "%Y-%m-%d"),
        from_date=datetime.datetime.strptime(getattr(args, 'from'), "%Y-%m-%d")
    )

    with open(path.join(args.modeldir, 'length-model.pkl'), 'rb') as fp:
        length_model = pickle.load(fp)

    with open(path.join(args.modeldir, 'distribution-model.pkl'), 'rb') as fp:
        dist_model = pickle.load(fp)

    with open(path.join(args.modeldir, 'gramma-model.pkl'), 'rb') as fp:
        gramma_model = pickle.load(fp)

    last_user_login = dict()

    for item in reader:
        user = item["who"]["id"]
        when = datetime.datetime.strptime(item["when"],
                                          "%Y-%m-%dT%H:%M:%S.%fZ")
        resource = item["subject"]["id"]
        ipaddress = item["who"]["attributes"]["ip-address"]
        ipaddress_string = (
            f' ({demo_ipaddress_helper[ipaddress]})'
            if ipaddress in demo_ipaddress_helper
            else ''
        )

        print(f'Checking "{user}" doing "{item["what"]["id"]}"')
        print(f'  time: {when}')
        print(f'  resource: {resource}')
        print(f'  ip-address: {ipaddress}{ipaddress_string}')

        print(f'  tests:')
        print(f'  - length: {check_length.validate(length_model, resource)}')
        print(f'  - distribution: {check_dist.validate(dist_model, resource)}')
        print(f'  - gramma: {check_gramma.validate(gramma_model, resource)}')

        if user in last_user_login:
            last_when, last_ipaddress = last_user_login[user]
            ipaddress_valid = check_ipaddress.validate(
                previuse_timestamp_ms=last_when.timestamp() * 1000,
                previuse_ipaddress=last_ipaddress,
                current_timestamp_ms=when.timestamp() * 1000,
                current_ipaddress=ipaddress
            )
            print(f'  - ipaddress: {ipaddress_valid}')
        else:
            print(f'  - ipaddress: {True}')
        print('')

        last_user_login[user] = (when, ipaddress)
