#!/usr/bin/python3
import os,time
print('DISCONNECT_WORKER_READY pid='+str(os.getpid()),flush=True)
while True:time.sleep(1)
