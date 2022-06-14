# coding: utf-8


def test_decide_downloader():
    from teilab.utils import decide_downloader

    assert decide_downloader("https://www.dropbox.com/sh/ID").__name__ == "Downloader"
    assert (
        decide_downloader("https://drive.google.com/u/0/uc?export=download&id=ID").__name__ == "GoogleDriveDownloader"
    )


def test_decide_extension():
    from teilab.utils import decide_extension

    assert decide_extension(content_encoding="image/png") == ".png"
    assert decide_extension(content_type="application/pdf") == ".pdf"
    assert decide_extension(content_encoding="image/webp", content_type="application/pdf") == ".webp"
    assert decide_extension(basename="hoge.zip") == ".zip"


def test_unzip():
    from teilab.utils import unzip

    # unzip("target.zip")
