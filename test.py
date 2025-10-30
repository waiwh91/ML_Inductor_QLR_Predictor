class ss():
    def __init__(self,a):
        self.a = a

test1 = ss(1)

def create(classtype):
    s =1
    a = classtype(s)

    print(isinstance(a, ss))

# print(isinstance(test1,ss))
create(ss)