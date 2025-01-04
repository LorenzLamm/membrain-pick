import typer
from click import Context
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False
)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    MemBrain-pick's data conversion / mesh projection module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    membrain_pick --help

    """

cli = typer.Typer(
    cls=OrderCommands,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)