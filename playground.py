from time import sleep
from utility import printProgress


for i in range(10):
    msg = f"workign on stuff{i}"
    printProgress(msg)
    sleep(0.2)