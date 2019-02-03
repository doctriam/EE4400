##############################################################################
# TITLE:  Numerical Computation in Deep Learning
# DESCRIPTION:  Basic python tests of numerical computation concepts
# AUTHOR:  Kenny Haynie
##############################################################################

from numpy import *
import sys,random

CURSOR_UP='\x1b[1A'
ERASE_LINE='\x1b[2K'


def main_menu():
    # Use data across all functions
    global data
    data=random.sample(range(10),3)

    # List of functions
    print('1. Softmax')
    print('0. Exit'),print()

    # User input
    menuOpt=int(input("Choose a number from the menu above:"))

    # Menu Options
    options = {1: softmax_func
              }

    # Clear menu and prompt
    for n in range(0,len(options)+3):
        sys.stdout.write(CURSOR_UP)
        sys.stdout.write(ERASE_LINE)
    print()

    # Run menu option
    if menuOpt in range(1,len(options)+1):
        options[menuOpt]()
    else:
        sys.exit()

    # Reload menu
    print(),print()
    main_menu()
    return

def softmax_func():
    print("---SOFTMAX---")
    numList=[exp(i) for i in data]
    den=sum(numList)
    pList=numList/den
    print("Dataset:"),print(data),print()
    print("Exponents:"),print(numList),print()
    print("Sum of Exponents:"),print(den),print()
    print("Softmax:")
    print ("[ %.6f, %.6f, %.6f ]" % (pList[0],pList[1],pList[2]))

main_menu()
