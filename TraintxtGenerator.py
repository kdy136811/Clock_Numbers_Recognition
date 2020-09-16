with open('train.txt', 'w') as text_file:
    for i in range(8000):
        filename = 'build/darknet/x64/data/clock/'+ str(i).zfill(5)
        print(filename+'.jpg', file=text_file)

with open('test.txt', 'w') as text_file:
    for i in range(8000,10000):
        filename = 'build/darknet/x64/data/clock/'+ str(i).zfill(5)
        print(filename+'.jpg', file=text_file)

with open('new_test.txt', 'w') as text_file:
    for i in range(11):
        filename = 'build/darknet/x64/data/their_clock/'+ str(i)
        print(filename+'.jpg', file=text_file)