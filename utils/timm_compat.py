"""Compatibility imports for both current and legacy timm releases."""

try:
    from timm.layers import DropPath, trunc_normal_
except ModuleNotFoundError as error:  # legacy timm has no top-level layers package
    if error.name != 'timm.layers':
        raise
    from timm.models.layers import DropPath, trunc_normal_

__all__ = ['DropPath', 'trunc_normal_']
