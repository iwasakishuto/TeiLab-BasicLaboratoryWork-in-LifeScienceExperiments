# coding: utf-8
def test_dict2str():
    from teilab.utils import dict2str
    assert dict2str({"key1":"val", "key2":1}) == 'key1=val, key2=1'
    assert dict2str({"key1":"val", "key2":1}, key_separator=":") ==  'key1:val, key2:1'
    assert dict2str({"key1":"val", "key2":1}, item_separator="🤔") == 'key1=val🤔key2=1'

def test_progress_reporthook_create():
    import urllib
    from teilab.utils import progress_reporthook_create
    # urllib.request.urlretrieve(url="hoge.zip", filename="hoge.zip", reporthook=progress_reporthook_create(filename="hoge.zip"))
    # hoge.zip	1.5%[--------------------] 21.5[s] 8.0[GB/s]	eta 1415.1[s]

def test_readable_bytes():
    from teilab.utils import readable_bytes
    units = ["","K","M","G","T","P","E","Z","Y"]
    for i,unit in enumerate(units):
        b,u = readable_bytes(2**(10*i))
        assert (b==1) and (u == unit+"B")

def test_verbose2print():
    from teilab.utils import verbose2print
    print_verbose = verbose2print(verbose=True)
    print_non_verbose = verbose2print(verbose=False)
    print_verbose("Hello, world.")
    # Hello, world.
    print_non_verbose("Hello, world.")