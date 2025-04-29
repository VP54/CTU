class Queue:
    def __init__(self, maxsize=10):
        self.head = None
        self.tail = None
        self.maxsize = maxsize
    
    def put(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def get(self):
        if self.head:
            last = self.head
            while last.next:
                last = last.next
            return last.data      

    def empty(self):
        # is queue empty?
        if self.head is None:
            return True
        else:
            return False
   
    def full(self):
        counter = 0
        if self.head:
            last = self.head
            while last.next:
                last = last.next
                counter +=1
                if counter == self.maxsize:
                    return True
                
        return False          
    
    def size(self):
        counter = 0
        if self.head:
            last = self.head
            while last.next:
                last = last.next
                counter +=1
        return counter


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.previous = None    

#more than full
q = Queue()
q.put('A')
q.put('F')
q.put('G')
q.put('L')
q.put('A')
q.put('F')
q.put('G')
q.put('L')
q.put('A')
q.put('F')
q.put('G')
q.put('L')

#empty
a = Queue()

#not empty not full
b = Queue()
b.put('A')
b.put('C')
b.put('G')
b.put('H')
b.put('A')
b.put('C')

def get_result(*objs):
    for obj in objs:
        print('--' * 20)
        print(f"Full queue: ")

        print(f"Gets first out element: \t {obj.get()}")
        print(f"Is queue full?: \t\t {obj.full()}")
        print(f"Is queue empty?: \t\t {obj.empty()}")
        print(f"What is the actual size?: \t {obj.size()}")
        print('--' * 20)

get_result(q, a, b)
