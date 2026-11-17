import os


def eclipse(i):
    oldfolder = os.getcwd()
    folder = f"MASTER{str(i + 1).zfill(5)}"
    os.chdir(folder)
    try:
        os.system("@eclrun eclipse MASTER0.DATA")
        print(f"Folder MASTER{i + 1} has been processed")
    finally:
        os.chdir(oldfolder)
