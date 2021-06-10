#coding: utf-8

def test_ask():
    from teilab.question import ask
    from teilab.utils import now_str
    ask(
        text=f"Testing Now ({now_str()})", 
        username="UiTei-Lab-Course", 
        icon_emoji=":rotating_light:"
    )