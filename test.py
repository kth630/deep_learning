#!/usr/bin/env python3
import pandas as pd

print(1)
frame = {}
num=0
for i in ['a', 'b', 'c', 'd', 'e']:
    frame[i] = [j for j in range(num, num+5)]
    num += 1
df = pd.DataFrame(frame)
print(df)