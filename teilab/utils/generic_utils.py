#coding: utf-8
import os
import sys
import time
import datetime
from typing import Tuple,Callable,Optional,Dict,Any,List
from numbers import Number

from ._exceptions import KeyError

def now_str(tz:Optional[datetime.timezone]=None, fmt:str="%Y-%m-%d@%H.%M.%S") -> str:
    """Returns new datetime string representing current time local to tz under the control of an explicit format string.

    Args:
        tz (Optional[datetime.timezone], optional) : Timezone object. If no ``tz`` is specified, uses local timezone. Defaults to ``None``.
        fmt (str, optional)                        : format string. See `Python Documentation <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_ Defaults to ``"%Y-%m-%d@%H.%M.%S"``.

    Returns:
        str: Formatted current time.

    Example:
        >>> from teilab.utils import now_str
        >>> now_str()
        '2021-06-01@11.22.14'
        >>> now_str(tz=datetime.timezone.utc)
        '2021-06-01@02.22.16'
        >>> now_str(fmt="%A, %d. %B %Y %I:%M%p")
        'Tuesday, 01. June 2021 11:22AM'
    """
    return datetime.datetime.now(tz=tz).strftime(fmt)

def readable_bytes(byte:Number) -> Tuple[Number,str]:
    """Unit conversion for readability.

    Args:
        byte (Number): File byte [B].

    Examples:
        >>> from teilab.utils import readable_bytes
        >>> for i in range(1,30,3):
        ...     byte = pow(10,i)
        ...     size, unit = readable_bytes(pow(10,i))
        ...     print(f"{byte:.1g}[B] = {size:.2f}[{unit}]")
        1e+01[B] = 10.00[B]
        1e+04[B] = 9.77[KB]
        1e+07[B] = 9.54[MB]
        1e+10[B] = 9.31[GB]
        1e+13[B] = 9.09[TB]
        1e+16[B] = 8.88[PB]
        1e+19[B] = 8.67[EB]
        1e+22[B] = 8.47[ZB]
        1e+25[B] = 8.27[YB]
        1e+28[B] = 8271.81[YB]
    """
    units = ["","K","M","G","T","P","E","Z","Y"]
    for unit in units:
        if (abs(byte)<1024.0) or (unit==units[-1]):
            break
        byte /= 1024.0 # size >> 10
    return (byte, unit+"B")

def progress_reporthook_create(filename:str="", bar_width:int=20, verbose:bool=True) -> Callable[[int,int,int], None]:
    """Create a progress reporthook for ``urllib.request.urlretrieve``        

    Args:
        filename (str, optional)  : Downloading filename.. Defaults to ``""``.
        bar_width (int, optional) : The width of progress bar. Defaults to ``20``.
        verbose (bool, optional)  : Whether to output the status. Defaults to ``True``.

    Returns:
        Callable[[int,int,int], None]: The ``reporthook`` which is a callable that accepts a ``block number``, a ``read size``, and the ``total file size`` of the URL target.

    Examples:
        >>> import urllib
        >>> from teilab.utils import progress_reporthook_create
        >>> urllib.request.urlretrieve(url="hoge.zip", filename="hoge.zip", reporthook=progress_reporthook_create(filename="hoge.zip"))
        hoge.zip	1.5%[--------------------] 21.5[s] 8.0[GB/s]	eta 1415.1[s]
    """
    def progress_reporthook_verbose(block_count:int, block_size:int, total_size:int) -> None:
        """``reporthook`` to report the current status.

        Args:
            block_count (int) : The number of block.
            block_size (int)  : The size of the data block.
            total_size (int)  : The size of the total data.
        """
        global _reporthook_start_time
        if block_count == 0:
            _reporthook_start_time = time.time()
            return
        progress_size = block_count*block_size
        percentage = min(1.0, progress_size/total_size)
        progress_bar = ("#" * int(percentage * bar_width)).ljust(bar_width, "-")
        
        duration = time.time() - _reporthook_start_time
        speed = progress_size / duration
        eta = (total_size-progress_size)/speed

        speed, speed_unit = readable_bytes(speed)
        
        sys.stdout.write(f"\r{filename}\t{percentage:.1%}[{progress_bar}] {duration:.1f}[s] {speed:.1f}[{speed_unit}/s] eta {eta:.1f}[s]")
        if progress_size >= total_size: print()
    def progress_reporthook_non_verbose(block_count:int, block_size:int, total_size:int) -> None:
        """``reporthook`` not to report the current status."""
        pass
    return progress_reporthook_verbose if verbose else progress_reporthook_non_verbose

def verbose2print(verbose:bool=True) -> callable:
    """Create a simple print function based on verbose
    
    Args:
        verbose (bool, optional) : Whether to print or not. Defaults to ``True``.

    Returns:
        callable: Print function

    Examples:
        >>> from teilab.utils import verbose2print
        >>> print_verbose = verbose2print(verbose=True)
        >>> print_non_verbose = verbose2print(verbose=False)
        >>> print_verbose("Hello, world.")
        Hello, world.
        >>> print_non_verbose("Hello, world.")    
    """
    if verbose:
        return print
    else:
        return lambda *args,**kwargs: None

def dict2str(d:Dict[Any,Any], item_separator:str=", ", key_separator:str="=") -> str:
    """Convert a dictionary to string.

    Args:
        d (Dict[Any,Any])              : A dictionary.
        item_separator (str, optional) : A separator between items. Defaults to ``", "``.
        key_separator (str, optional)  : A separator between keys. Defaults to ``"="``.

    Returns:
        str: A textual summary of the contents of the dictionary.

    Examples:
        >>> from teilab.utils import dict2str
        >>> dict2str({"key1":"val", "key2":1})
        'key1=val, key2=1'
        >>> dict2str({"key1":"val", "key2":1}, key_separator=":")
        'key1:val, key2:1'
        >>> dict2str({"key1":"val", "key2":1}, item_separator="🤔")
        'key1=val🤔key2=1'
    """
    return str(item_separator).join([f"{k}{key_separator}{v}" for k,v in d.items()])

def check_supported(lst:List[Any], **kwargs):
    """Check whether all ``kwargs.values()`` in the ``lst``.

    Args:
        lst (List[Any]) : candidates for each value in ``kwargs.values()``.
        kwargs          : ``key`` is the varname that is easy to understand when an error occurs

    Examples:
        >>> from teilab.utils import check_supported
        >>> check_supported(lst=range(3), val=1)
        >>> check_supported(lst=range(3), val=100)
        KeyError: Please choose the argment 'val' from ['0', '1', '2'], but you chose '100'
        >>> check_supported(lst=range(3), val1=1, val2=2)
        >>> check_supported(lst=range(3), val1=1, val2=100)
        KeyError: Please choose the argment 'val2' from ['0', '1', '2'], but you chose '100'

    Raise:
        KeyError: If ``kwargs.values()`` not in the ``lst``
    """
    for k,v in kwargs.items():
        if v not in lst:
            lst = ', '.join([f"'{e}'" for e in lst])
            raise KeyError(f"Please choose the argment '{k}' from [{lst}], but you chose '{v}'")