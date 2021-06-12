# coding: utf-8
import os
import re
import urllib
import zipfile
import requests
from tqdm import tqdm
from typing import Optional,Dict,List,Tuple

from .generic_utils import now_str
from .generic_utils import readable_bytes
from .generic_utils import progress_reporthook_create
from .generic_utils import verbose2print

CONTENT_ENCODING2EXT:Dict[str,str] = {
    "gzip"                      : ".gz",
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
    "text/html"                             : ".txt",
}

def unzip(path:str, verbose:bool=True) -> Tuple[str,List[str]]:
    """Unzip a zipped file ( Only support the file with ``.zip`` extension. )

    Args:
        path (str)               : The path to zipped file.
        verbose (bool, optional) : Whether to print verbose or not. Defaults to ``True``.

    Returns:
        Tuple[str,List[str]]: The directory where the expanded data is stored and the List of their respective file paths.

    Examples:
        >>> from teilab.utils import unzip
        >>> unzip("target.zip")
    """
    extracted_file_paths = []
    print = verbose2print(verbose=verbose)
    root,ext = os.path.splitext(path)
    if ext not in [".zip", ".gz"]:
        print(f"Do not support to extract files with the '{ext}' extension.")
    else:
        if not os.path.exists(root):
            os.mkdir(root)
        print("[Unzip] Show file contents:")
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                info.filename = info.orig_filename.encode('cp437').decode('utf-8')
                if (os.sep!="/") and (os.sep in info.filename):
                    info.filename = info.filename.replace(os.sep, "/")
                z.extract(member=info, path=root)
                extracted_file_path = os.path.join(root, info.filename)
                extracted_file_paths.append(extracted_file_path)
                print(f"\t* {info.filename}")
    return root, extracted_file_paths

def decide_extension(content_encoding:Optional[str]=None, content_type:Optional[str]=None, basename:Optional[str]=None):
    """Decide File Extension based on ``content_encoding`` and ``content_type``

    Args:
        content_encoding (Optional[str], optional) : The MIME type of the resource or the data.
        content_type (Optional[str], optional)     : The Content-Encoding entity header is used to compress the media-type.
        basename (Optional[str], optional)         : The basename.

    Returns:
        str: The file extension which starts with "."

    Examples:
        >>> from teilab.utils import decide_extension
        >>> decide_extension(content_encoding="image/png")
        '.png'
        >>> decide_extension(content_type="application/pdf")
        '.pdf'
        >>> decide_extension(content_encoding="image/webp", content_type="application/pdf")
        '.webp'
        >>> decide_extension(basename="hoge.zip")
        '.zip'
    """
    ext = CONTENT_ENCODING2EXT.get(content_encoding, CONTENT_TYPE2EXT.get(content_type, os.path.splitext(str(basename))[-1]))
    return ext     

class Downloader():
    """General Downloader"""
    @classmethod
    def download_file(cls, url:str, dirname:str=".", basename:str="", path:Optional[str]=None, verbose:bool=True, expand:bool=True, **kwargs) -> str:
        """Download a file and expand it if you want.

        Args:
            url (str)                      : The URL of the file you want to download.
            dirname (str, optional)        : The directory where downloaded data will be saved. Defaults to ``"."``.
            basename (str, optional)       : The basename of the target file. Defaults to ``""``.
            path (Optional[str], optional) : Where and what name to save the downloaded file. Defaults to ``None``.
            verbose (bool, optional)       : Whether to print verbose or not. Defaults to ``True``.
            expand (bool, optional)        : Whether to expand the downloaded file. Defaults to ``True``

        Returns:
            path (str) : The path to the downloaded file.
        
        Examples:
            >>> import os
            >>> from teilab.utils import Downloader
            >>> path = Downloader.download_file(url="http://ui-tei.rnai.jp/")
            [Download] URL: http://ui-tei.rnai.jp/
            * Content-Encoding : None
            * Content-Length   : 32.1 [KB]
            * Content-Type     : text/html
            * Save Destination : ./2021-06-01@21.30.html
            ===== Progress =====
            2021-06-01@21.30.04	100.0%[####################] 0.0[s] 1.3[MB/s] eta -0.0[s]
            Do not support to extract files with the '.html' extension.
            >>> os.path.exists(path)
            True
        """
        path = cls.download_target_file(url=url, dirname=dirname, basename=basename, path=path, verbose=True, **kwargs)
        if expand:
            path, extracted_file_paths = unzip(path=path, verbose=verbose)
        return path

    @staticmethod
    def download_target_file(url:str, dirname:str=".", basename:str=".", path:Optional[str]=None, bar_width:int=20, verbose:bool=True, **kwargs) -> str:
        """Download the target file.

        Args:
            url (str)                      : The URL of the file you want to download.
            dirname (str, optional)        : The directory where downloaded data will be saved. Defaults to ``"."``.
            basename (str, optional)       : The basename of the target file. Defaults to ``""``.
            path (Optional[str], optional) : Where and what name to save the downloaded file. Defaults to ``None``.
            bar_width (int, optional)      : The width of progress bar. Defaults to ``20``.
            verbose (bool, optional)       : Whether to print verbose or not. Defaults to ``True``.

        Returns:
            path (str) : The path to the downloaded file.
        
        Examples:
            >>> import os
            >>> from teilab.utils import Downloader
            >>> path = Downloader.download_target_file(url="http://ui-tei.rnai.jp/")
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
            with urllib.request.urlopen(url) as web_file:
                headers = dict(web_file.headers._headers)
            filename, path = Downloader.prepare_for_download(url=url, basename=os.path.basename(url), dirname=dirname, path=path, headers=headers, verbose=verbose)
            if verbose: print("===== Progress =====")
            _, res = urllib.request.urlretrieve(url=url, filename=path, reporthook=progress_reporthook_create(filename=filename, bar_width=bar_width, verbose=verbose))
        except urllib.error.URLError:
            print(f"[URLError] Please check if the URL is correct, given {url}")
        except Exception as e:
            print(f"[{e.__class__.__name__}] {e}")
        return path

    @staticmethod
    def prepare_for_download(url:str="", dirname:str=".", basename:str="", path:Optional[str]=None, headers:Optional[Dict[str,str]]=None, verbose:bool=True) -> Tuple[str,str]:
        """Get Information from webfile header and prepare for downloading.

        Args:
            url (str, optional)                         : The URL of the file you want to download. Defaults to ``""``.
            dirname (str, optional)                     : The directory where downloaded data will be saved. Defaults to ``"."``.
            basename (str, optional)                    : The basename of the target file. Defaults to ``""``.
            path (Optional[str], optional)              : Where and what name to save the downloaded file. Defaults to ``None``.
            headers (Optional[Dict[str,str]], optional) : The header information of the target file. Defaults to ``{}``.
            verbose (bool, optional)                    : Whether to print verbose or not. Defaults to ``True``.

        Returns:
            Tuple[str,str]: ``filename`` and ``path`` of the file that will be downloaded.

        Examples:
            >>> from teilab.utils import Downloader
            >>> filename, path = Downloader.prepare_for_download(
            ...     url="http://ui-tei.rnai.jp/",
            ...     basename="index.html",
            ...     dirname=".",
            ...     path=None,
            >>> )
            [Download] URL: http://ui-tei.rnai.jp/
            * Content-Encoding : None
            * Content-Length   : 32.1 [KB]
            * Content-Type     : text/html
            * Save Destination : ./index.html
            >>> filename, path
            ('index.html', './index.html')
        """
        # Get the information of the file you want to download from the header.
        if headers is None:
            with urllib.request.urlopen(url) as web_file:
                headers = dict(web_file.headers._headers)
        content_encoding = headers.get("Content-Encoding")
        content_length   = "{0:.1f} [{1}]".format(*readable_bytes(int(headers.get("Content-Length", 0))))
        content_type     = headers.get("Content-Type").split(";")[0]
        # Decide the download destination
        if basename=="": 
            basename = now_str()
        if path is None:
            root, _ = os.path.splitext(basename)
            guessed_ext = decide_extension(content_encoding, content_type, basename)
            filename = root+guessed_ext
            path = os.path.join(dirname, filename)
        else:
            filename = os.path.split(path)[-1]
        # Show the results.
        if verbose: print(
            f"[Download] URL: {url}",
            f"* Content-Encoding : {content_encoding}",
            f"* Content-Length   : {content_length}",
            f"* Content-Type     : {content_type}",
            f"* Save Destination : {path}",
            sep="\n"
        )
        return (filename, path)

class GoogleDriveDownloader(Downloader):
    """Specific Downloader for files in GoogleDrive"""
    CHUNK_SIZE = 32768
    DRIVE_URL  = "https://docs.google.com/uc?export=download"

    @staticmethod
    def prepare_for_download(url:str="", dirname:str=".", basename:str="", path:Optional[str]=None, headers:Optional[Dict[str,str]]=None, verbose:bool=True, driveId:Optional[str]=None) -> Tuple[str,str]:
        if driveId is None:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id")
            if len(q)==0:
                raise TypeError("Please specify the target Google Drive Id using ``url`` or ``driveId`` arguments.")
            else:
                driveId=q[0]
        if basename=="":
            basename = driveId
        # Start a Session
        params = {"id":driveId}
        session = requests.Session()
        response = session.get(url=GoogleDriveDownloader.DRIVE_URL, params=params, stream=True)
        for key,val in response.cookies.items():
            if key.startswith("download_warning"):
                params.update({"confirm":val})
                break
        # Get Information from headers
        headers = session.head(url=GoogleDriveDownloader.DRIVE_URL, params=params).headers
        return [*Downloader.prepare_for_download(
            url=url,
            dirname=dirname,
            basename=basename,
            path=path,
            headers=headers,
            verbose=verbose,
        ), session, params]


    @staticmethod
    def download_target_file(url:str, dirname:str=".", basename:str="", path:Optional[str]=None, driveId:Optional[str]=None, verbose:bool=True, **kwargs) -> str:
        """Download the target Google Drive file.

        Args:
            url (str)                         : The URL of the file you want to download.
            dirname (str, optional)           : The directory where downloaded data will be saved. Defaults to ``"."``.
            basename (str, optional)          : The basename of the target file. Defaults to ``""``.
            path (Optional[str], optional)    : Where and what name to save the downloaded file. Defaults to ``None``.
            driveId (Optional[str], optional) : The GoogleDrive's file ID. Defaults to ``None``.
            verbose (bool, optional)          : Whether to print verbose or not. Defaults to ``True``.

        Raises:
            TypeError: When Google Drive File ID is not detected from ``driveId`` and ``url`` .

        Returns:
            str: The path to the downloaded file.        
        """
        filename, path, session, params = GoogleDriveDownloader.prepare_for_download(url=url, basename=basename, dirname=dirname, path=path, verbose=verbose)
        # Get contents
        response = session.get(GoogleDriveDownloader.DRIVE_URL, params=params, stream=True)
        with open(path, "wb") as f:
            with tqdm(response.iter_content(GoogleDriveDownloader.CHUNK_SIZE), desc=driveId) as pbar:
                for i,chunk in enumerate(pbar, start=1):
                    if chunk:
                        f.write(chunk)
                        pbar.set_postfix({"Downloaded": "{0:.1f} [{1}]".format(*readable_bytes(i*GoogleDriveDownloader.CHUNK_SIZE))})
        return path

def decide_downloader(url:str) -> Downloader:
    """Decide ``Downloader`` from ``url``

    Args:
        url (str): The URL of the file you want to download.

    Returns:
        Downloader: File Downloader for target ``url``.

    Examples:
        >>> from teilab.utils import decide_downloader
        >>> decide_downloader("https://www.dropbox.com/sh/ID").__name__
        'Downloader'
        >>> decide_downloader("https://drive.google.com/u/0/uc?export=download&id=ID").__name__
        'GoogleDriveDownloader'
    """
    url_domain = re.match(pattern=r"^https?:\/\/(.+?)\/", string=url).group(1)
    return {
        "drive.google.com" : GoogleDriveDownloader,
    }.get(url_domain, Downloader)    
