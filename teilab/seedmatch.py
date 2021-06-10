#coding: utf-8
import re
import requests
import pandas as pd

from .utils._config import SEEDMATCH_URL

def get_matched_mRNAs(seedseq:str) -> pd.DataFrame:
    """Get a list of mRNAs with ``seedseq`` in 3'UTR

    Args:
        seedseq (str): A query sequence.

    Returns:
        pd.DataFrame: A data frame with a column named ``"SystematicName"`` meaning Accession numbers for each mRNA and a column named ``"NumHits"`` meaning how many ``seedseq`` sequences are in its 3'UTR

    Examples:
        >>> from teilab.seedmatch import get_matched_mRNAs
        >>> df_matched_mRNAs = get_matched_mRNAs(seedseq="gagttca")
        >>> print(df_matched_mRNAs.to_markdown())
        |      | SystematicName   |   NumHits |
        |-----:|:-----------------|----------:|
        |    0 | NM_001004713     |         1 |
        |    1 | NM_173860        |         1 |
        |    2 | NM_001005493     |         1 |
        |    : |     :            |         : |
        | 3643 | NM_015139        |         1 |
        | 3644 | NM_015463        |         1 |
        | 3645 | NM_007189        |         1 |
        >>> from teilab.utils import now_str
        >>> now_str()
        '2021-06-10@23.36.12'

    You can also get the data with the command like ``curl`` .

    .. code-block:: shell
    
        $ curl -d "seedseq=gagttca" <SEEDMATCH_URL>

    """
    ret = requests.post(url=SEEDMATCH_URL, data={"seedseq": seedseq.lower()})
    return pd.DataFrame(
        data=re.findall(pattern=r"\n(.+)\t([0-9]+)", string=ret.text),
        columns=["SystematicName", "NumHits"]
    )