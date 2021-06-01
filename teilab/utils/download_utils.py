# coding: utf-8
import os
import urllib
from typing import Optional,Dict
from tqdm import tqdm
from pathlib import Path

from .generic_utils import now_str
from .generic_utils import readable_bytes
from .generic_utils import progress_reporthook_create

CONTENT_ENCODING2EXT:Dict[str,str] = {
    "x-gzip"                    : ".gz",
    "image/jpeg"                : ".jpg",
    "image/jpx"                 : ".jpx", 
    "image/png"                 : ".png",
    "image/gif"                 : ".gif",
    "image/webp"                : ".webp",
    "image/x-canon-cr2"         : ".cr2",
    "image/tiff"                : ".tif",
    "image/bmp"                 : ".bmp",
    "image/vnd.ms-photo"        : ".jxr",
    "image/vnd.adobe.photoshop" : ".psd",
    "image/x-icon"              : ".ico",
    "image/heic"                : ".heic",
}

CONTENT_TYPE2EXT:Dict[str,str] = {
    "application/epub+zip"                  : ".epub",
    "application/zip"                       : ".zip",
    "application/x-tar"                     : ".tar",
    "application/x-rar-compressed"          : ".rar",
    "application/gzip"                      : ".gz",
    "application/x-bzip2"                   : ".bz2",
    "application/x-7z-compressed"           : ".7z",
    "application/x-xz"                      : ".xz",
    "application/pdf"                       : ".pdf",
    "application/x-msdownload"              : ".exe",
    "application/x-shockwave-flash"         : ".swf",
    "application/rtf"                       : ".rtf",
    "application/octet-stream"              : ".eot",
    "application/postscript"                : ".ps",
    "application/x-sqlite3"                 : ".sqlite",
    "application/x-nintendo-nes-rom"        : ".nes",
    "application/x-google-chrome-extension" : ".crx",
    "application/vnd.ms-cab-compressed"     : ".cab",
    "application/x-deb"                     : ".deb",
    "application/x-unix-archive"            : ".ar",
    "application/x-compress"                : ".Z",
    "application/x-lzip"                    : ".lz",
    "text/html"                             : ".html",
}

def decide_extension(content_encoding:Optional[str]=None, content_type:Optional[str]=None, filename:Optional[str]=None):
    """Decide File Extension based on ``content_encoding`` and ``content_type``

    Args:
        content_encoding (str) : The MIME type of the resource or the data.
        content_type (str)     : The Content-Encoding entity header is used to compress the media-type.
        filename (str)         : The filename.

    Returns:
        ext (str): The file extension which starts with "."

    Examples:
        >>> from teilab.utils import decide_extension
        >>> decide_extension(content_encoding="image/png")
        '.png'
        >>> decide_extension(content_type="application/pdf")
        '.pdf'
        >>> decide_extension(content_encoding="image/webp", content_type="application/pdf")
        '.webp'
        >>> decide_extension(filename="hoge.zip")
        '.zip'
    """
    ext = CONTENT_ENCODING2EXT.get(content_encoding, CONTENT_TYPE2EXT.get(content_type, os.path.splitext(str(filename))[-1]))
    return ext

def download_file(url:str, dirname:str=".", path:Optional[str]=None, bar_width:int=20, verbose:bool=True) -> str:
    """Download a file.

    Args:
        url (str)       : File URL.
        dirname (str)   : The directory where downloaded data will be saved.
        path (str)      : path/to/downloaded_file
        bar_width (int) : The width of progress bar.
        verbose (bool)  : Whether print verbose or not.

    Returns:
        path (str) : path/to/downloaded_file
    
    Examples:
        >>> import os
        >>> from teilab.utils import download_file
        >>> path = download_file(url="http://ui-tei.rnai.jp/")
        [Download] URL: http://ui-tei.rnai.jp/
        * Content-Encoding : None
        * Content-Length   : 31.8 [KB]
        * Content-Type     : text/html
        * Save Destination : ./2021-06-01@11.26.html
        ===== Progress =====
        2021-06-01@11.26.48	100.0%[####################] 0.0[s] 1.0[MB/s]	eta -0.0[s]
        >>> os.path.exists(path)
        True
    """    
    try:
        # Get Information from webfile header
        with urllib.request.urlopen(url) as web_file:
            headers = dict(web_file.headers._headers)
        content_encoding = headers.get("Content-Encoding")
        content_length   = "{0:.1f} [{1}]".format(*readable_bytes(int(headers.get("Content-Length", 0))))
        content_type     = headers.get("Content-Type").split(";")[0]
        filename = os.path.basename(url)
        if len(filename)==0: 
            filename = now_str()
        if path is None:
            root, _ = os.path.splitext(filename)
            guessed_ext = decide_extension(content_encoding, content_type, filename)
            path = os.path.join(dirname, root+guessed_ext)
        if verbose:
            print(
                f"[Download] URL: {url}",
                f"* Content-Encoding : {content_encoding}",
                f"* Content-Length   : {content_length}",
                f"* Content-Type     : {content_type}",
                f"* Save Destination : {path}",
                "===== Progress =====",
                sep="\n"
            )
        _, res = urllib.request.urlretrieve(url=url, filename=path, reporthook=progress_reporthook_create(filename=filename, bar_width=bar_width, verbose=verbose))
    except urllib.error.URLError:
        print(f"[URLError] Please check if the URL is correct, given {url}")
    except Exception as e:
        print(f"[{e.__class__.__name__}] {e}")
    return path
