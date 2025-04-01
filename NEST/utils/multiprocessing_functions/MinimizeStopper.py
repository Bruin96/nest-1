import time

class TimeoutReturnError(Exception):
   
    # Constructor or Initializer
    def __init__(self, value):
        self.value = value
   
    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.value))

class MinimizeStopper(object):
	def __init__(self, max_sec=60):
		self.max_sec = max_sec
		self.start = time.time()
	def __call__(self, xk=None):
		elapsed = time.time() - self.start
		if elapsed > self.max_sec:
			raise TimeoutReturnError(xk)
