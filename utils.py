from random import choice, randint
from string import ascii_uppercase

## Generates a random string of X characters (defaults to 5)
def rand_string(length=5):
    return ''.join(choice(ascii_uppercase) for i in range(length))

def rand_num(maxdigits=5):
    endint = (10**maxdigits)-1
    return randint(0,endint)