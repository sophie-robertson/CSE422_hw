import hashlib
import numpy as np

l = 6
b = 256

def hash(x, i, seed):
    input = str(x) + str(seed)
    hex_hash = hashlib.md5(input.encode("utf-8")).hexdigest()
    byte = hex_hash[2*i:2*(i+1)]
    return int(byte, 16)

class CountMinSketch:

    def __init__(self, cons):
        self.data = np.zeros((l,b))
        self.cons = cons # determines which update rule to use
    
    def inc(self, x, s):
        for i in range(l):
            h = hash(x, i, s)
            self.data[i][h] += 1
    
    def inc_cons(self, x, s):
        min = self.count(x, s)
        for i in range(l):
            h = hash(x, i, s)
            val = self.data[i][h]
            self.data[i][h] = val + 1 if val == min else val

    def count(self, x, s):
        min = -1
        for i in range(l):
            h = hash(x, i, s)
            count = self.data[i][h]
            if (count < min or min == -1):
                min = count
        return min
    
    def run(self, input, s):
        for elt in input:
            if not self.cons:
                self.inc(elt, s)
            else:
                self.inc_cons(elt, s)
    
    def clear(self):
        self.data = np.zeros((l,b))


# count min sketch executions with different stream types
n = 150
size = (2*(n**3) + 9*(n**2) - 5*n) / 6

# automatically creates in heavy-last order
def create_dataset(): 
    data = []
    for i in range(n+1, n**2 + 1):
        data.append(i)
    for i in range(1, n+1):
        for j in range(i**2):
            data.append(i)
    return np.array(data)

# if cons, apply conservative update rule
def heavy_first(cons):
    print("Heavy first")
    data = np.flip(create_dataset())
    test(data, cons)

def heavy_last(cons):
    print("Heavy last")
    data = create_dataset()
    test(data, cons)

def random(cons):
    print("Random")
    data = create_dataset()
    np.random.shuffle(data)
    test(data, cons)

# runs 10 iterations of inputs into CMS and returns (1) average
# frequency for 100, and (2) average number of heavy hitters
def test(data, cons):
    count_100 = 0
    count_hh = 0
    for i in range(10):
        CMS = CountMinSketch(cons)
        CMS.run(data, i)

        # update frequency/heavy hitter counts for iteration
        count_100 += CMS.count(100, i)
        for elt in range(1,n**2 + 1):
            CMS.count(elt, i)
            if (CMS.count(elt, i) >= size / 100):
                count_hh += 1
        CMS.clear()
    
    avg_100 = count_100 / 10
    avg_hh = count_hh / 10
    print(f"Average Frequency for 100: {avg_100}")
    print(f"Average number of heavy hitters: {avg_hh}")
    print()

def main():
    print("Normal CMS")
    heavy_first(False)
    heavy_last(False)
    random(False)

    print("Conservative CMS")
    heavy_first(True)
    heavy_last(True)
    random(True)

if __name__ == '__main__':
    main()