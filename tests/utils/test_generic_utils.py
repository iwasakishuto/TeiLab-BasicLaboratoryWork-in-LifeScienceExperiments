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
    for i in range(1,30,3):
        byte = pow(10,i)
        size, unit = readable_bytes(pow(10,i))
        print(f"{byte:.1g}[B] = {size:.2f}[{unit}]")
    # 1e+01[B] = 10.00[B]
    # 1e+04[B] = 9.77[KB]
    # 1e+07[B] = 9.54[MB]
    # 1e+10[B] = 9.31[GB]
    # 1e+13[B] = 9.09[TB]
    # 1e+16[B] = 8.88[PB]
    # 1e+19[B] = 8.67[EB]
    # 1e+22[B] = 8.47[ZB]
    # 1e+25[B] = 8.27[YB]
    # 1e+28[B] = 8271.81[YB]

def test_verbose2print():
    from teilab.utils import verbose2print
    print_verbose = verbose2print(verbose=True)
    print_non_verbose = verbose2print(verbose=False)
    print_verbose("Hello, world.")
    # Hello, world.
    print_non_verbose = verbose2print("Hello, world.")    

