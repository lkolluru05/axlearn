import logging

class Adder:
  def __init__(self, increment):
    logging.info("Adder created")
    self.increment = increment

  def __del__(self):
    logging.info("Adder destroyed")

  def add(self, x):
    return x + self.increment