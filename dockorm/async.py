# encoding: utf-8
"""
Utilities for asynchronously launching docker containers.
"""
from __future__ import unicode_literals
try:
    # Python3
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    from futures.thread import ThreadPoolExecutor

from functools import partial
from docker import Client
from requests_futures.sessions import FuturesSession
from six import (
    iteritems,
)
from tornado import gen


class AsyncDockerClient(object):
    """
    Async wrapper around docker.Client that returns futures on all methods by
    running them on a thread pool.
    """
    def __init__(self, executor=None, **client_kwargs):
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
        self._executor = executor
        self._client = Client(**client_kwargs)

    def __getattr__(self, name):
        '''Creates a function, based on docker_client.name that returns a
        Future. If name is not a callable, returns the attribute directly.
        '''
        fn = getattr(self._client, name)

        # Make sure it really is a function first
        if not callable(fn):
            return fn

        def method(*args, **kwargs):
            return self._executor.submit(fn, *args, **kwargs)

        return method


def coroutine(f):
    """
    Mark a function as a coroutine for use with sync_coroutine or
    gen.coroutine.

    Does nothing by itself but set a flag used by @synchronous_coroutines and
    @asynchronous_coroutines decorators.
    """
    f.__is_coroutine = True
    return f


def sync_coroutine(f):
    """
    Synchronous version of gen.coroutine.
    """
    def f_as_coroutine(*args, **kwargs):
        co = f(*args, **kwargs)
        try:
            value = gen.maybe_future(next(co)).result()
            while True:
                value = gen.maybe_future(co.send(value)).result()
        except gen.Return as e:
            return e.value
        except StopIteration:
            return None
    return f_as_coroutine


def _specialize_coroutines(cls, func):
    """
    Traverse a newly-created class looking for functions marked as coroutines
    and apply `func` to them.
    """
    overrides = {}
    for supercls in cls.__mro__:
        for key, value in iteritems(supercls.__dict__):
            if getattr(value, '__is_coroutine', False):
                if key in overrides:
                    continue
                overrides[key] = func(value)
    for key, value in iteritems(overrides):
        setattr(cls, key, value)
    return cls


# Actual decorators for public use.
synchronous_coroutines = partial(_specialize_coroutines, func=sync_coroutine)
asynchronous_coroutines = partial(_specialize_coroutines, func=gen.coroutine)
