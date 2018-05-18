
import typing
import datetime
from .request import trail_request


def trail_read(from_date: datetime.datetime, to_date: datetime.datetime,
               page_size: int=25) -> typing.Iterator[typing.Any]:
    page_number = 0

    while True:
        page_number += 1
        response = trail_request('GET', '/trails', parameters={
            'from': from_date.isoformat(timespec='microseconds') + 'Z',
            'to': to_date.isoformat(timespec='microseconds') + 'Z',
            'pageSize': page_size,
            'page': page_number,
            'sort': 'when'
        })

        yield from response

        if len(response) < page_size:
            break
