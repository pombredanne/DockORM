# encoding: utf-8
"""
Container class.
"""
from __future__ import print_function, unicode_literals
from itertools import chain
import json
from subprocess import call
import sys

from docker import Client
from docker.utils import kwargs_from_env
from six import (
    iteritems,
    iterkeys,
    itervalues,
    text_type,
)
from tornado import gen

from IPython.utils.py3compat import string_types

from IPython.utils.traitlets import (
    Any,
    Dict,
    HasTraits,
    Instance,
    List,
    Type,
    Unicode,
    TraitError,
)

from .py3compat_utils import strict_map
from .async import (
    AsyncDockerClient,
    coroutine,
    synchronous_coroutines,
    asynchronous_coroutines,
)

def print_build_output(build_output):
    success = True
    for raw_message in build_output:
        message = json.loads(raw_message)
        try:
            print(message['stream'], end="")
        except KeyError:
            success = False
            print(message['error'])
    return success


def scalar(l):
    """
    Get the first and only item from a list.
    """
    assert len(l) == 1
    return l[0]


class ContainerBase(HasTraits):
    """
    A specification for creation of a container.
    """

    organization = Unicode()

    def _organization_changed(self, name, old, new):
        if new and not new.endswith('/'):
            self.organization = new + '/'

    image = Unicode()

    def _image_changed(self, name, old, new):
        if not new:
            raise TraitError("Must supply a value for image.")

    tag = Unicode(default_value='latest')

    def full_imagename(self, tag=None):
        return '{}{}:{}'.format(
            self.organization,
            self.image,
            tag or self.tag,
        )

    name = Unicode()

    def _name_default(self):
        return self.image + '-running'

    build_path = Unicode()

    links = List(Instance(__name__ + '.Link'))

    def format_links(self):
        return {}

    volumes_readwrite = Dict()

    volumes_readonly = Dict()

    @property
    def volume_mount_points(self):
        """
        Volumes are declared in docker-py in two stages.  First, you declare
        all the locations where you're going to mount volumes when you call
        create_container.

        Returns a list of all the values in self.volumes or
        self.read_only_volumes.
        """
        return list(
            chain(
                itervalues(self.volumes_readwrite),
                itervalues(self.volumes_readonly),
            )
        )

    @property
    def volume_binds(self):
        """
        The second half of declaring a volume with docker-py happens when you
        actually call start().  The required format is a dict of dicts that
        looks like:

        {
            host_location: {'bind': container_location, 'ro': True}
        }
        """
        volumes = {
            key: {'bind': value, 'ro': False}
            for key, value in iteritems(self.volumes_readwrite)
        }
        ro_volumes = {
            key: {'bind': value, 'ro': True}
            for key, value in iteritems(self.volumes_readonly)
        }
        volumes.update(ro_volumes)
        return volumes

    ports = Dict(help="Map from container port -> host port.")

    @property
    def open_container_ports(self):
        return strict_map(int, iterkeys(self.ports))

    @property
    def port_bindings(self):
        out = {}
        for key, value in self.ports:
            if isinstance(ports, (list, tuple)):
                key = '/'.join(strict_map(text_type, key))
            out[key] = value
        return out

    environment = Dict()

    # This should really be something like:
    # Either(Instance(str), List(Instance(str)))
    command = Any()

    client = Instance(Client, kw=kwargs_from_env())

    def build(self, tag=None, display=True, rm=True):
        """
        Build the container.

        If display is True, write build output to stdout.  Otherwise return it
        as a generator.

        Building asynchronously is not supported.
        """
        output = gen.maybe_future(
            self.client.build(
                self.build_path,
                self.full_imagename(tag=tag),
                # This is in line with the docker CLI, but different from
                # docker-py's default.
                rm=rm,
            )
        ).result()
        if display:
            print_build_output(output)
            return None
        else:
            return list(output)

    @coroutine
    def run(self, command=None, tag=None, attach=False, rm=False):
        """
        Run this container.
        """
        if rm and not attach:
            raise ValueError(
                "Auto-remove is not supported with detached execution."
            )

        container = yield self.client.create_container(
            self.full_imagename(tag),
            name=self.name,
            ports=self.open_container_ports,
            volumes=self.volume_mount_points,
            detach=not attach,
            stdin_open=attach,
            tty=attach,
            command=command,
            environment=self.environment,
        )

        yield self.client.start(
            container,
            binds=self.volume_binds,
            port_bindings=self.ports,
            links=self.format_links(),
        )

        if attach:
            call(['docker', 'attach', self.name])
            if rm:
                yield self.client.remove_container(self.name)

    def _matches(self, container):
        return '/' + self.name in container['Names']

    @coroutine
    def instances(self, all=True):
        """
        Return any instances of this container, running or not.
        """
        raise gen.Return(
            [
                c for c in (yield self.client.containers(all=all))
                if self._matches(c)
            ]
        )

    @coroutine
    def running(self):
        """
        Return the running instance of this container, or None if no container
        is running.
        """
        container = yield self.instances(all=False)
        if container:
            raise gen.Return(scalar(container))
        else:
            raise gen.Return(None)

    @coroutine
    def stop(self):
        raise gen.Return(self.client.stop(self.name))

    @coroutine
    def purge(self, stop_first=True, remove_volumes=False):
        """
        Purge all containers of this type.
        """
        for container in (yield self.instances()):
            if stop_first:
                yield self.client.stop(container)
            else:
                yield self.client.kill(container)
            yield self.client.remove_container(
                container,
                v=remove_volumes,
            )

    @coroutine
    def inspect(self, tag=None):
        """
        Inspect any running instance of this container.
        """
        raise gen.Return(
            (
                yield self.client.inspect_container(
                    self.name,
                )
            )
        )

    @coroutine
    def images(self):
        """
        Return any images matching our current organization/name.

        Does not filter by tag.
        """
        raise gen.Return(
            (yield self.client.images(self.full_imagename().split(':')[0]))
        )

    @coroutine
    def remove_images(self):
        """
        Remove any images matching our current organization/name.

        Does not filter by tag.
        """
        for image in (yield self.images()):
            yield self.client.remove_image(image)

    @coroutine
    def logs(self, all=False):
        out = []
        for container in (yield self.instances(all=all)):
            out.append(
                {
                    'Id': container,
                    'Logs': (yield self.client.logs(container)),
                }
            )
        raise gen.Return(out)

    @coroutine
    def join(self):
        """
        Wait until there are no instances of this container running.
        """
        container = yield self.running()
        if container:
            yield self.client.wait(container)


@synchronous_coroutines
class Container(ContainerBase):
    """
    Synchronously-executing container.
    """
    pass


@asynchronous_coroutines
class AsyncContainer(Container):
    """
    Asynchronously-executing container.
    """
    client = Instance(AsyncDockerClient, kw=kwargs_from_env())


class Link(HasTraits):
    """
    A link between containers.
    """
    container = Instance(Container)
    alias = Unicode()
