
import json
import typing
import urllib.request as request
import urllib.parse as parse


def trail_request(method, path, parameters: typing.Dict={},
                  body: typing.Any=None):
    url = parse.ParseResult(
        scheme='http',
        netloc='localhost:8080',
        path=path,
        query=parse.urlencode(parameters),
        params='',
        fragment=''
    )

    req = request.Request(
        parse.urlunparse(url),
        headers={
            'accept': 'application/json',
            'Content-Type': 'application/json'
        },
        method=method,
        data=None if body is None else json.dumps(body).encode()
    )

    with request.urlopen(req) as f:
        if f.status == 204:
            response = []
        else:
            response = json.load(f)

        if f.status >= 400:
            raise RuntimeError(f'{response.message}, '
                               f'reasons: {response.reasons}')

    return response
