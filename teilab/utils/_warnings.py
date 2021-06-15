#coding: utf-8
import os
from typing import Tuple

from ._path import REPO_DIR
from ..__meta__ import __documentation__

__all__ = [
    "TeiLabImprementationWarning",
    "_pack_warning_args",
    "InsufficientUnderstandingWarning",
]

class TeiLabImprementationWarning(Warning):
    """ 
    - Warnings that developers will resolve. 
    - Developers are now solving in a simple stupid way. (will be replaced.)
    """

_sep = "[_]"
def _pack_warning_args(*args):
    return _sep.join([str(e) for e in args])

class InsufficientUnderstandingWarning(Warning):
    """Warning if you are using the module without understanding.
    
    Examples:
        >>> import sys
        >>> import warnings
        >>> from teilab.utils._warnings import InsufficientUnderstandingWarning, _pack_warning_args
        >>> warnings.warn(_pack_warning_args("message", __file__, sys._getframe().f_code.co_name), category=InsufficientUnderstandingWarning)
    """
    def __init__(self, args:str=_pack_warning_args("", "index.html", "")):
        filename = "index.html"
        co_name  = ""
        args = args.split(_sep)
        if len(args)==1:
            message = args[0]
        elif len(args)==2:
            message, filename = args
        else:
            message, filename, co_name = args[:3]
        self.message = message
        if os.path.exists(filename):
            filename = os.path.relpath(path=os.path.abspath(filename), start=REPO_DIR).replace("/", ".").replace(".py", f".html#{co_name}")
        self.url = f"{__documentation__}/{filename}"

    def __str__(self):
        return f"""
It seems that you do not understand the underlying principles correctly. Please check the documentation (URL: {self.url} )
[Message from Author]
{self.message}
"""