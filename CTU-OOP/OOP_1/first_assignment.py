class FizzBuzz():
  """ Trida pro ukol s nazvem Fizz Buzz """

  def __init__(self, number: int):
    self.number = number
  
  def podminky(self, number):
      """ Funkce ktera kontroluje podminky hry """

      if (number %3 == 0) and (number % 5 == 0): 
          print("Fizz Buzz")
      elif number % 5 == 0: 
          print("Buzz")
      elif number % 3 == 0: 
          print("Fizz")
      else: 
          print(number)
        

  def main(self):
      """ Hlavni funkce, ktera generuje radu cisel a vola podminky """

      for i in range(self.number):
        #print(i)
        self.podminky(i)


FizzBuzz = FizzBuzz(32)

FizzBuzz.main()

