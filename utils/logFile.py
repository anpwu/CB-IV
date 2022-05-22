def log(logfile,str,out=True):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    if out:
        print(str)