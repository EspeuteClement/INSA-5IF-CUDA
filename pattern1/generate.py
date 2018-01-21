from random import shuffle

with open("input.txt","w") as f :
    f.write("" + str((255 * (256) / 2) * 10)  + "\n")
    a = []
    index = 0
    for i in range(0,256) :
        for j in range(0,i) :
            for k in range(0,10) :
                a.append(i) 
    shuffle(a);
    for v in a :
        f.write(str(v) + "\n");
