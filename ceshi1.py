import re

m = '0FFJze5UOO_index056.jpg'

match = re.search(r'_index(\d+)', m)
index = match.group(1)
index_list = [int(digit) for digit in index]
print(index_list)