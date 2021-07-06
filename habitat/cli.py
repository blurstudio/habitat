import click
from . import Resolver
import logging
import os

logger = logging.getLogger(__name__)


# No one wants to actually type `--help`
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


class SharedSettings(object):
    def __init__(self, configs=None, distros=None, verbosity=0, script_output=None):
        self.verbosity = verbosity
        self.script_output = os.path.abspath(script_output or ".")
        self.config_paths = configs
        self.distro_paths = distros
        self._resolver = None

    @property
    def resolver(self):
        if self._resolver is None:
            self._resolver = Resolver(self.config_paths, self.distro_paths)
        return self._resolver


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c",
    "--configs",
    multiple=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help="glob paths to find configuration.",
)
@click.option(
    "-d",
    "--distros",
    multiple=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help="glob paths to find distro configuration.",
)
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    count=True,
    help="Increase the verbosity of the output. Can be used up to 3 times.",
)
@click.option(
    "--script-output",
    type=click.Path(dir_okay=False, resolve_path=True),
    help="The commands that generate an OS specific script to configure the"
    "environment will write to this location.",
)
@click.pass_context
def cli(ctx, configs, distros, verbosity, script_output):
    ctx.obj = SharedSettings(configs, distros, verbosity)
    if verbosity > 2:
        verbosity = 2
    level = [logging.WARNING, logging.INFO, logging.DEBUG][verbosity]
    logging.basicConfig(level=level)


@cli.command()
@click.argument("uri")
@click.pass_obj
def env(settings, uri):
    """Configures and launches a new shell with the resolved setup."""
    logger.info("Context: {}".format(uri))
    ret = settings.resolver.resolve(uri)
    click.echo(ret.dump())
    # TODO: Generate a temp script to write to
    ret.write_script(r"c:\temp\hab.bat")


@cli.command()
@click.argument("uri")
@click.pass_obj
def dump(settings, uri):
    """Resolves and prints the requested setup."""
    logger.info("Context: {}".format(uri))
    ret = settings.resolver.resolve(uri)
    click.echo(ret.dump())


@cli.command()
@click.argument("uri")
@click.pass_obj
def activate(settings, uri):
    """Resolves the setup and updates in the current shell."""
    click.echo("Not implemented")


if __name__ == "__main__":
    cli()
