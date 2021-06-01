# coding: utf-8
import os
import re
import urllib
import zipfile
import requests
from tqdm import tqdm
from typing import Optional,Dict,List,Tuple
from pathlib import Path

from ._config import GAS_WEBAPP_URL
from ._path import DATA_DIR
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
    "text/html"                             : ".html",
}

def unzip(path:str, verbose:bool=True) -> List[str]:
    """Unzip a zipped file ( Only support the file with ``.zip`` extension. )

    Args:
        path (str)               : The path to zipped file.
        verbose (bool, optional) : Whether to print verbose or not. Defaults to ``True``.

    Returns:
        List[str]: Paths to extracted files.

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
        with zipfile.ZipFile(path) as compressed_f:
            for name in compressed_f.namelist():
                compressed_f.extract(name, path=root)
                extracted_file_path = os.path.join(root, name)
                extracted_file_paths.append(extracted_file_path)
                print(f"\t* {name}")
    return extracted_file_paths

def decide_extension(content_encoding:Optional[str]=None, content_type:Optional[str]=None, filename:Optional[str]=None):
    """Decide File Extension based on ``content_encoding`` and ``content_type``

    Args:
        content_encoding (Optional[str], optional) : The MIME type of the resource or the data.
        content_type (Optional[str], optional)     : The Content-Encoding entity header is used to compress the media-type.
        filename (Optional[str], optional)         : The filename.

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
        >>> decide_extension(filename="hoge.zip")
        '.zip'
    """
    ext = CONTENT_ENCODING2EXT.get(content_encoding, CONTENT_TYPE2EXT.get(content_type, os.path.splitext(str(filename))[-1]))
    return ext     

class Downloader():
    """General Downloader"""
    @classmethod
    def download_file(cls, url:str, dirname:str=".", path:Optional[str]=None, verbose:bool=True, expand:bool=True, **kwargs) -> str:
        """Download a file and expand it if you want.

        Args:
            url (str)                      : The URL of the file you want to download.
            dirname (str, optional)        : The directory where downloaded data will be saved. Defaults to ``"."``.
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
        path = cls.download_target_file(url=url, dirname=dirname, path=path, verbose=True, **kwargs)
        if expand:
            unzip(path=path, verbose=verbose)
        return path

    @staticmethod
    def download_target_file(url:str, dirname:str=".", path:Optional[str]=None, bar_width:int=20, verbose:bool=True, **kwargs) -> str:
        """Download the target file.

        Args:
            url (str)                      : The URL of the file you want to download.
            dirname (str, optional)        : The directory where downloaded data will be saved. Defaults to ``"."``.
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
            filename, path = Downloader.prepare_for_download(url=url, filename=os.path.basename(url), dirname=dirname, path=path, headers=headers, verbose=verbose)
            if verbose: print("===== Progress =====")
            _, res = urllib.request.urlretrieve(url=url, filename=path, reporthook=progress_reporthook_create(filename=filename, bar_width=bar_width, verbose=verbose))
        except urllib.error.URLError:
            print(f"[URLError] Please check if the URL is correct, given {url}")
        except Exception as e:
            print(f"[{e.__class__.__name__}] {e}")
        return path

    @staticmethod
    def prepare_for_download(url:str="", filename:str="", dirname:str=".", path:Optional[str]=None, headers:Dict[str,str]={}, verbose:bool=True) -> Tuple[str,str]:
        """Get Information from webfile header and prepare for downloading.

        Args:
            url (str, optional)               : The URL of the file you want to download. Defaults to ``""``.
            filename (str, optional)          : The filename of the target file. Defaults to ``""``.
            dirname (str, optional)           : The directory where downloaded data will be saved. Defaults to ``"."``.
            path (Optional[str], optional)    : Where and what name to save the downloaded file. Defaults to ``None``.
            headers (Dict[str,str], optional) : The header information of the target file. Defaults to ``{}``.
            verbose (bool, optional)          : Whether to print verbose or not. Defaults to ``True``.

        Returns:
            Tuple[str,str]: ``filename`` and ``path`` of the file that will be downloaded.

        Examples:
            >>> from teilab.utils import Downloader
            >>> filename, path = Downloader.prepare_for_download(
            ...     url="http://ui-tei.rnai.jp/",
            ...     filename="index.html",
            ...     dirname=".",
            ...     path=None,
            ...     headers={"Content-Length": "32874", 'Content-Type': 'text/html; charset=UTF-8'}
            >>> )
            [Download] URL: http://ui-tei.rnai.jp/
            * Content-Encoding : None
            * Content-Length   : 32.1 [KB]
            * Content-Type     : text/html
            * Save Destination : ./index.html
            >>> filename, path
            ('index.html', './index.html')
        """
        content_encoding = headers.get("Content-Encoding")
        content_length   = "{0:.1f} [{1}]".format(*readable_bytes(int(headers.get("Content-Length", 0))))
        content_type     = headers.get("Content-Type").split(";")[0]
        if filename=="": 
            filename = now_str()
        if path is None:
            root, _ = os.path.splitext(filename)
            guessed_ext = decide_extension(content_encoding, content_type, filename)
            path = os.path.join(dirname, root+guessed_ext)
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
    def download_target_file(url:str, dirname:str=".", path:Optional[str]=None, driveId:Optional[str]=None, verbose:bool=True, **kwargs) -> str:
        """Download the target Google Drive file.

        Args:
            url (str)                         : The URL of the file you want to download.
            dirname (str, optional)           : The directory where downloaded data will be saved. Defaults to ``"."``.
            path (Optional[str], optional)    : Where and what name to save the downloaded file. Defaults to ``None``.
            driveId (Optional[str], optional) : The GoogleDrive's file ID. Defaults to ``None``.
            verbose (bool, optional)          : Whether to print verbose or not. Defaults to ``True``.

        Raises:
            TypeError: When Google Drive File ID is not detected from ``driveId`` and ``url`` .

        Returns:
            str: The path to the downloaded file.        
        """
        if driveId is None:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id")
            if len(q)==0:
                raise TypeError("Please specify the target Google Drive Id using ``url`` or ``driveId`` arguments.")
            else:
                driveId=q[0]
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
        filename, path = GoogleDriveDownloader.prepare_for_download(url=url, filename=driveId, dirname=dirname, path=path, headers=headers, verbose=verbose)
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

def get_teilab_data(password:str, verbose:bool=True) -> str:
    """Get data which is necessary for this class.

    Args:
        password (str)           : Password. (Because some data are ubpublished.)
        verbose (bool, optional) : Whether print verbose or not. Defaults to ``True``.

    Returns:
        str: The path to the downloaded file.

    Examples:
        >>> from teilab.utils import get_teilab_data
        >>> path = get_teilab_data(password="PASSWORD")
        Try to get data from <SECRET_URL>
        This is our unpublished data, so please treat it confidential.
        [Download] URL: <SECRET_URL>
        * Content-Encoding : None
        * Content-Length   : 45.9 [MB]
        * Content-Type     : application/zip
        * Save Destination : PATH/TO/PASSWORD.zip
        ===== Progress =====
        <SECRET FILENAME>	100.0%[####################] 45.3[s] 1.0[MB/s]	eta -0.0[s]
        Save data at PATH/TO/PASSWORD.zip
        [Unzip] Show file contents:
            * <SECRET_FILE_1>
            * <SECRET_FILE_2>
            * :
            * <SECRET_FILE_N>
        >>> path
        'PATH/TO/PASSWORD.zip'

    Below is the code for the GAS(Google Apps Script) API server.

    .. code-block:: js

        const P = PropertiesService.getScriptProperties();
        const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(P.getProperty("sheetname"))
        const values = sheet.getRange("A2:C").getValues();

        var Password2dataURL = {};
        for (let i=0; i<values.length; i++){
          Password2dataURL[values[i][0]] = values[i].slice(1);
        }

        function doPost(e) {
          var response = {message: "Invalid Password", dataURL:""};
          var password = e.parameter.password;

          if (password in Password2dataURL){
            let data = Password2dataURL[password]
            response.dataURL = data[0]
            response.message = data[1]
          }

          var output = ContentService.createTextOutput();
          output.setMimeType(ContentService.MimeType.JSON);
          output.setContent(JSON.stringify(response));
          return output;
        }
    """
    print = verbose2print(verbose=verbose)
    # Get the target data URL.
    ret = requests.post(url=GAS_WEBAPP_URL, data={"password": password})
    data = ret.json()
    dataURL = data.get("dataURL", "")
    message = data.get("message", "")
    print(f"Try to get data from {dataURL}\n{message}")
    # Use the specific ``Downloader`` to download the target data.
    downloader = decide_downloader(url=dataURL)
    path = downloader.download_file(url=dataURL, path=os.path.join(DATA_DIR, f"{password}.zip"), verbose=verbose, expand=True)
    print(f"Save data at {path}")
    return path