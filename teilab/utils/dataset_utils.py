#coding: utf-8
import os
import pandas as pd
import requests
from pathlib import Path
from typing import List

from ._config import GAS_WEBAPP_URL
from ._path import DATA_DIR, SAMPLE_LIST_PATH
from .download_utils import decide_downloader
from .generic_utils import verbose2print

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

def get_filelists() -> List[Path]:
    """Get the path list of files used in the lecture.

    Returns:
        List[Path]: The path lists for datasets.

    Examples:
        >>> from teilab.utils import get_filelists
        >>> filelists = get_filelists()
        >>> len(filelists)
        13
        >>> filelists[0].name
        'US91503671_253949442637_S01_GE1_105_Dec08_1_1.txt'
    """
    filenames = pd.read_csv(SAMPLE_LIST_PATH).FileName.to_list()
    p = Path(DATA_DIR)
    return sorted([path for path in p.glob("**/*.txt") if path.name in filenames])
