# int, int -> string
# Given integer num and base b, converts num to a string representation in base b
def convert(num, b):
    quotient = num//b                     #find initial quotient
    remainder = num % b                   #find initial remainder
    string = "0123456789ABCDEF"           #string for conversion based off of given conversions for bases >10

    #Base Case
    if quotient == 0:                     #base case

        return string[remainder]          #go to string and return the index position that is converted

    #Recursive Call
    while quotient != 0:                  #algorithm stops when quotient is 0 so keep going while not equal to 0

         return convert(quotient, b) + string[remainder]                  #call function again with inputs if quotient is still not 0 and then add to index position of remainder without needing to reverse the string