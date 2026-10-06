"""Sample process RSS; CUDA allocator peaks are collected separately."""
import threading
import time
import psutil
import os
from pathlib import Path


class RSSMonitor:
    def __init__(self,interval=0.02):
        try:
            self.process=psutil.Process()
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            self.process=None
        self.interval=interval
        self.peak=self.read();self.stop=threading.Event()

    def read(self):
        if self.process is not None:return self.process.memory_info().rss
        # Some container runtimes expose /proc/self but remap os.getpid().
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')

    def __enter__(self):
        def sample():
            while not self.stop.wait(self.interval):
                self.peak=max(self.peak,self.read())
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.peak=max(self.peak,self.read())
        self.stop.set();self.thread.join()
