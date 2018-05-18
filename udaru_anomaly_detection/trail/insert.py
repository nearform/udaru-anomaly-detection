
import typing
import datetime
from .request import trail_request


def trail_insert(when: datetime.datetime, who: typing.Any=None,
                 what: typing.Any=None, subject: typing.Any=None,
                 where: typing.Any=None, why: typing.Any=None,
                 meta: typing.Any=None) -> None:
    trail_request('POST', '/trails', body={
        'when': when.isoformat(timespec='microseconds') + 'Z',
        'who': {} if who is None else who,
        'what': {} if what is None else what,
        'subject': {} if subject is None else subject,
        'where': {} if where is None else where,
        'why': {} if why is None else why,
        'meta': {} if meta is None else meta
    })
