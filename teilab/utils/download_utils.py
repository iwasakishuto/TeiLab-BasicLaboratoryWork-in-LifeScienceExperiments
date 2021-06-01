# coding: utf-8
import os
import sys
import urllib
import zipfile
import requests
from tqdm import tqdm
from typing import Optional,Dict,List
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
    """
    print = verbose2print(verbose=verbose)
    root,ext = os.path.splitext(path)
    if not os.path.exists(root):
        os.mkdir(root)

    print("[Unzip] Show file contents:")
    extracted_file_paths = []
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

def download_google_drive_file(url:str, driveId:Optional[str]=None, dirname:str=".", path:Optional[str]=None, bar_width:int=20, verbose:bool=True) -> str:
    CHUNK_SIZE = 32768
    DRIVE_URL  = "https://docs.google.com/uc?export=download"
    if driveId is None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id")
        if len(q)==0:
            raise TypeError("Please specify the target Google Drive Id using ``url`` or ``driveId`` arguments.")
        else:
            driveId=q[0]
    # Start a Session
    params = {"id":driveId}
    session = requests.Session()
    response = session.get(url=DRIVE_URL, params=params, stream=True)
    for key,val in response.cookies.items():
        if key.startswith("download_warning"):
            params.update({"confirm":val})
            break
    # Get Information from headers
    headers = session.head(url=DRIVE_URL, params=params).headers
    content_encoding = headers.get("Content-Encoding")
    content_length   = "{0:.1f} [{1}]".format(*readable_bytes(int(headers.get("Content-Length", 0))))
    content_type     = headers.get("Content-Type").split(";")[0]
    if path is None:
        guessed_ext = decide_extension(content_encoding, content_type)
        path = os.path.join(dirname, driveId+guessed_ext)
    print(
        f"[Download] Drive ID: {driveId}",
        f"* Content-Encoding : {content_encoding}",
        f"* Content-Length   : {content_length}",
        f"* Content-Type     : {content_type}",
        f"* Save Destination : {path}",
        "===== Progress =====",
        sep="\n"
    )
    # Get contents
    response = session.get(DRIVE_URL, params=params, stream=True)
    with open(path, "wb") as f:
        for i,chunk in enumerate(response.iter_content(CHUNK_SIZE), start=1):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                sys.stdout.write("\r{0}\t{1:.1f} [{2}]".format(driveId, *readable_bytes(i*CHUNK_SIZE)))
    return path        


def download_file(url:str, dirname:str=".", path:Optional[str]=None, bar_width:int=20, verbose:bool=True) -> str:
    """Download a file.

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
        print = verbose2print(verbose=verbose)
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

def get_teilab_data(password:str, verbose:bool=True) -> str:
    """Get data which is necessary for this class.

    Args:
        password (str)                 : Password. (Because some data are ubpublished.)
        verbose (bool, optional)       : Whether print verbose or not. Defaults to ``True``.

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
    ret = requests.post(url=GAS_WEBAPP_URL, data={"password": password})
    data = ret.json()
    dataURL = data.get("dataURL", "")
    message = data.get("message", "")
    print(f"Try to get data from {dataURL}\n{message}")
    path = download_file(url=dataURL, path=os.path.join(DATA_DIR, f"{password}.zip"), verbose=verbose)
    print(f"Save data at {path}")
    if os.path.splitext(path)[-1] == ".zip":
        _ = unzip(path=path, verbose=verbose)
    return path