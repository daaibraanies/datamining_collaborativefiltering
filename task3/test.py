data = {
    'a':{0:1,1:20,4:-90,5:60,6:12,7:14},
    'b':{0:100,1:220,4:-90,5:6,6:12,7:124},
    'c':{0:-1,1:20,4:-90,5:60,6:1002,7:14},
    'd':{0:1,1:-20,4:-900,5:6450,6:12,7:114}
}


max_value = -float('inf')  #минус бесконечность
indices = {}

aa = data.values()

for d in data.values():
    for a in d:
        print(a)



print("Максимальное значение {}б расположено в data[{}][{}]".format(
    max_value,indices[0],indices[1]
))