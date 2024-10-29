import os

def GetCurrentDir():
    currentDir = os.path.dirname(os.path.abspath(__file__))
    return currentDir