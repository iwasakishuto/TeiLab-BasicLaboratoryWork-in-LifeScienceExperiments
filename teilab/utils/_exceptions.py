#coding: utf-8

class KeyError(KeyError):
    """Overwrite original KeyError so that coloring can be used when outputting an error message"""
    def __str__(self):
        return ', '.join(self.args)