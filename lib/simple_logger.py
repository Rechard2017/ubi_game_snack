import os
import datetime


class simple_logger(object):
    def __init__(self, _flush=True):
        self.flush = _flush
        pass
    def info(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        print(s, flush=self.flush)
    def debug(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        print(s, flush=self.flush)
    def error(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        print(s, flush=self.flush)
    def warn(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        print(s, flush=self.flush)


class file_logger(object):
    def __init__(self, name, flush=True):
        self.file = open(name, 'w')
        self.name = name
        self.size = 0
        self.flush = flush
    def __del__(self):
        self.file.close()
        if self.size == 0:
            os.remove(self.name)
    def info(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        self.file.write(s + '\n')
        self.file.flush()
        print(s, flush=self.flush)
        self.size += len(s) + 1
    def debug(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        self.file.write(s + '\n')
        print(s, flush=self.flush)
        self.size += len(s) + 1
    def error(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        self.file.write(s + '\n')
        print(s, flush=self.flush)
        self.size += len(s) + 1
    def warn(self, s):
        s = datetime.datetime.strftime(datetime.datetime.now(),'[%Y%m%d-%H:%M:%S] ') + s
        self.file.write(s + '\n')
        print(s, flush=self.flush)
        self.size += len(s) + 1

