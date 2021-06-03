#coding: utf-8
import os
import pandas as pd
import requests
from pathlib import Path
from typing import List,Union,Optional

from .utils._config import GAS_WEBAPP_URL
from .utils._path import DATA_DIR, SAMPLE_LIST_PATH
from .utils.download_utils import decide_downloader
from .utils.generic_utils import verbose2print

class Samples():
    """Utility Sample Class for this lecture.
    
    Attributes:
        df_ (pd.DataFrame)        : Sample information described in text file at ``sample_list_path`` .
        SampleNumber (List[str])  : Index numbers for each sample.
        FileName (List[str])      : File namse for each sample.
        Condition (List[str])     : Experimental conditions for each sample.

    Examples:
        >>> from teilab.datasets import Samples
        >>> from teilab.utils._path import SAMPLE_LIST_PATH
        >>> samples = Samples(sample_list_path=SAMPLE_LIST_PATH)
        >>> samples.__dict__.keys()
        dict_keys(['df_', 'SampleNumber', 'FileName', 'Condition'])
    """
    def __init__(self, sample_list_path:str):
        self.df_ = pd.read_csv(sample_list_path)
        for col in self.df_.columns:
            setattr(self, col, self.df_[col].tolist())
        self.grouping()
    
    def grouping(self):
        self.FileName


class TeiLabDataSets():
    """Utility Datasets Class for this lecture.

    Args:
        verbose (bool, optional) : Whether print verbose or not. Defaults to ``True``.

    Attributes:
        verbose (bool)   : Whether print verbose or not. Defaults to ``True``.
        print (callable) : Print function.
        sample (Samples) : Datasts Samples. 
        root (Path)      : Root Directory for Datasets. ( ``DATA_DIR`` )

    Examples:
        >>> from teilab.datasets import TeiLabDataSets
        >>> datasets = TeiLabDataSets(verbose=False)
        There are not enough datasets. Use ``.get_data`` to prepare all the required datasets.
        >>> datasets.satisfied
        False
        >>> datasets.get_data(password="PASSWORD1")
        >>> datasets.get_data(password="PASSWORD2")
        >>> datasets.satisfied
        True
    """
    TARGET_GeneName:str = "VIM"             #: TARGET_GeneName (str) ``GeneName`` of the target RNA (vimentin)
    TARGET_SystematicName:str = "NM_003380" #: TARGET_SystematicName (str) ``SystematicName`` of the target RNA (vimentin)
    META_COLNAMES:List[str] = ["FeatureNum","ControlType","ProbeName","GeneName","SystematicName"] #: META_COLNAMES (List[str]) Column names for metadata.
    DATA_COLNAMES:List[str] = ["gProcessedSignal"]                                                 #: DATA_COLNAMES (List[str]) Column names for expression data.
    def __init__(self, verbose:bool=True):
        self.verbose = verbose
        self.print:callable = verbose2print(verbose=verbose)
        self.init()

    def init(self):
        """Initialization"""
        self.samples:Samples = Samples(SAMPLE_LIST_PATH)
        self.root:Path = Path(DATA_DIR)
        if not self.satisfied:
            self.print(f"There are not enough datasets. Use ``.get_data`` to prepare all the required datasets.")
    
    def get_data(self, password:str) -> str:
        """Get data which is necessary for this lecture.

        Args:
            password (str) : Password. (Because some data are ubpublished.)

        Returns:
            str: The path to the downloaded file.

        Examples:
            >>> from teilab.utils import TeiLabDataSets
            >>> datasets = TeiLabDataSets()
            >>> path = datasets.get_teilab_data(password="PASSWORD")
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
        # Get the target data URL.
        path = ""
        ret = requests.post(url=GAS_WEBAPP_URL, data={"password": password})
        data = ret.json()
        dataURL = data.get("dataURL", "")
        message = data.get("message", "")
        if len(dataURL)==0:
            self.print(f"Could not get the valid URL.\n{message}")
        else:
            self.print(f"Try to get data from {dataURL}\n{message}")
            path = os.path.join(DATA_DIR, f"{password}.zip")
            if os.path.exists(path):
                self.print(f"Data already exists, so do nothing here.")
            else:
                # Use the specific ``Downloader`` to download the target data.
                downloader = decide_downloader(url=dataURL)
                path = downloader.download_file(url=dataURL, path=path, verbose=self.verbose, expand=True)
                self.print(f"Saved data at {path}")
        return path

    def get_filePaths(self) -> List[Path]:
        """Get the path list of files used in the lecture.

        Returns:
            List[Path]: The path lists for datasets.

        Examples:
            >>> from teilab.utils import TeiLabDataSets
            >>> datasets = TeiLabDataSets()
            >>> filelists = datasets.get_filePaths()
            >>> len(filelists)
            13
            >>> filelists[0].name
            'US91503671_253949442637_S01_GE1_105_Dec08_1_1.txt'
        """
        return sorted([path for path in self.root.glob("**/*.txt") if path.name in self.samples.FileName], key=lambda x:self.samples.FileName.index(x.name))

    @property
    def filePaths(self) -> List[Path]:
        """The path lists for datasets."""
        return self.get_filePaths()

    @property
    def satisfied(self) -> bool:
        """Whether to get all necessary data or not."""
        return len(self.filePaths) == len(self.samples.FileName)

    def read(self, no:Union[int,str,List[int]], sep:Optional[str]="\t", header:Union[int,List[int]]="infer", nrows:Optional[int]=None, usecols:Optional[Union[List[str],callable]]=None, **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Read sample(s) data as ``pd.DataFrame``

        Args:
            no (Union[int,str,List[int]])                           : Target sample number(s) or ``"all"`` .
            sep (Optional[str], optional)                           : Delimiter to use. Defaults to ``"\\t"``
            header (Union[int,List[int]], optional)                 : Row number(s) to use as the column names, and the start of the data. Defaults to ``"infer"``.
            nrows (Optional[int], optional)                         : Number of rows of file to read. Useful for reading pieces of large files. Defaults to ``None``.
            usecols (Optional[Union[List[str],callable]], optional) : Return a subset of the columns. Defaults to ``None``.
            **kwargs (dict)                                         : Other keyword arguments for ``pd.read_csv`` .

        Raises:
            TypeError: When argument ``no`` is an instance of unexpected type or is an unexpected value.

        Returns:
            Union[pd.DataFrame, List[pd.DataFrame]]: DataFrame of the specified sample(s).

        Examples:
            >>> from teilab.datasets import TeiLabDataSets
            >>> datasets = TeiLabDataSets(verbose=False)
            >>> dfs = datasets.read(no="all", header=9)
            >>> len(dfs)
            13
            >>> type(dfs[0])
            pandas.core.frame.DataFrame
        """
        kwargs.update({"sep":sep, "header":header, "nrows":nrows,"usecols":usecols})
        if isinstance(no, list):
            return [self._read_csv(no=n, **kwargs) for n in no]
        elif isinstance(no, str):
            if no=="all":
                return [self._read_csv(no=n, **kwargs) for n in range(len(self.filePaths))]
            else:
                raise TypeError("Please specify the sample number.")
        else:
            return self._read_csv(no=no, **kwargs)

    def _read_csv(self, no:int, **kwargs) -> pd.DataFrame:
        """Read the sample as ``pd.DataFrame``        

        Args:
            no (int)      : Target sample number.
            kwargs (dict) : Keyword arguments for ``pd.read_csv`` .

        Returns:
            pd.DataFrame: DataFrame of the specified sample.

        Examples:
            >>> from teilab.datasets import TeiLabDataSets
            >>> datasets = TeiLabDataSets(verbose=False)
            >>> df = datasets._read_csv(no=0, header=9)
            >>> len(df)
            62976
            >>> type(df)
            pandas.core.frame.DataFrame
        """
        filepath = self.filePaths[no]
        self.print(f"Read data from '{filepath.relative_to(self.root)}'")
        return pd.read_csv(filepath_or_buffer=filepath, **kwargs)

    def read_data(self, no:Union[int,str,List[int]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Read sample(s) 'expression' data as ``pd.DataFrame``

        Args:
            no (Union[int,str,List[int]]) : Target sample number(s) or ``"all"`` .

        Returns:
            Union[pd.DataFrame, List[pd.DataFrame]]: DataFrame of the specified sample(s) 'expression' data.

        Examples:
            >>> from teilab.datasets import TeiLabDataSets
            >>> datasets = TeiLabDataSets(verbose=False)
            >>> dfs = datasets.read_data(no=[0,1,2])
            >>> len(dfs)
            3
            >>> len(dfs[0])
            62976
        """
        return self.read(no=no, header=9)

    def read_meta(self, no:Union[int,str,List[int]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Read sample(s) 'meta' data as ``pd.DataFrame``

        Args:
            no (Union[int,str,List[int]]) : Target sample number(s) or ``"all"`` .

        Returns:
            Union[pd.DataFrame, List[pd.DataFrame]]: DataFrame of the specified sample(s) 'meta' data.

        Examples:
            >>> from teilab.datasets import TeiLabDataSets
            >>> datasets = TeiLabDataSets(verbose=False)
            >>> dfs = datasets.read_meta(no="all")
            >>> len(dfs)
            13
            >>> len(dfs[0])
            1
        """
        return self.read(no=no, header=1, nrows=1)

    def read_summary(self, no:Union[int,str,List[int]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Read sample(s) 'summary' data as ``pd.DataFrame``

        Args:
            no (Union[int,str,List[int]]) : Target sample number(s) or ``"all"`` .

        Returns:
            Union[pd.DataFrame, List[pd.DataFrame]]: DataFrame of the specified sample(s) 'summary' data.

        Examples:
            >>> from teilab.datasets import TeiLabDataSets
            >>> datasets = TeiLabDataSets(verbose=False)
            >>> dfs = datasets.read_summary(no=[0,3,8,9])
            >>> len(dfs)
            4
            >>> len(dfs[0])
            1
        """
        return self.read(no=no, header=5, nrows=1)

    @staticmethod
    def reliable_filter(df:pd.DataFrame, name:Optional[str]=None) -> pd.DataFrame:
        """Create a dataframe which means whether data is reliable or not.

        Args:
            df (pd.DataFrame)              : Input dataframe.
            name (Optional[str], optional) : The column name. Defaults to ``None``.

        Returns:
            pd.DataFrame: Filter DataFrame which means whether data is reliable or not.

        Examples:
            >>> import pandas as pd
            >>> from teilab.datasets import TeiLabDataSets
            >>> datasets = TeiLabDataSets(verbose=False)
            >>> df_sg = datasets.read_data(0)
            >>> len(df_sg), datasets.reliable_filter(df_sg).sum().values[0]
            (62976, 30385)
            >>> df_us = datasets.read_data(-1)
            >>> len(df_us), datasets.reliable_filter(df_us).sum().values[0]
            (62976, 23434)
        """
        control   = df.ControlType      == 0
        present   = df.gIsPosAndSignif  == 1
        uniform   = df.gIsFeatNonUnifOL == 0
        wellabove = df.gIsWellAboveBG   == 1
        saturated = df.gIsSaturated     == 0
        popnol    = df.gIsFeatPopnOL    == 0
        return pd.DataFrame(
            data=control & present & uniform & wellabove & saturated & popnol,
            columns=name,
        )