from Data_Generator import Data
BLOCK_SIZE = [64, 64, 64]
path = '/home/bxsh/Liver_data'

data = Data(path, BLOCK_SIZE)
count = 0
while True:
    count += 1
    try:
        x, y = data.next()
        print x.shape, y.shape
    except Exception as e:
        data = Data(path, BLOCK_SIZE)
