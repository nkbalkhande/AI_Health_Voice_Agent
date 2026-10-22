from typing import TYPE_CHECKING

__all__ = ["ProviderFactory", "settings"]

if TYPE_CHECKING:
    from backend.core.config import settings
    from backend.core.factory import ProviderFactory


# Resolved on attribute access: importing any backend.core submodule runs this
# module first, so eagerly pulling in the factory made backend.core.exceptions
# depend on backend.application.chat, which imports backend.core.exceptions back.
def __getattr__(name: str):
    if name == "settings":
        from backend.core.config import settings

        return settings
    if name == "ProviderFactory":
        from backend.core.factory import ProviderFactory

        return ProviderFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
