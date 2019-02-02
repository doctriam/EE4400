##############################################################################
# TITLE:  Deep Learning - Linear Algebra
# DESCRIPTION:  Going through the linear algebra concepts associated with the
#               Deep Learning directed study
# AUTHOR:  Kenny Haynie
##############################################################################

from numpy import *
import sys

CURSOR_UP='\x1b[1A'
ERASE_LINE='\x1b[2K'

matrix33=matrix([[1,2,3],[4,5,6],[7,8,9]])
matrix33T=matrix33.T
matrix33I=matrix33.I

def main_menu():
    global matrix33,matrix33T,matrix33I
    # List of functions
    print("1. Vectors, Matrices, and Tensors")
    print("2. Transpose a Matrix")
    print("3. Matrix Addition")
    print("4. Matrix Scalar Addition")
    print("5. Matrix Multiplication")
    print("6. Inverse Matrix")
    print("0. Exit")

    # User input
    menuOption=int(input("Choose a number from the menu above:"))

    # Menu options
    options = {1: basic_Matrix,
               2: matrix_transpose,
               3: matrix_addition,
               4: matrix_scalar_addition,
               5: matrix_multiplication,
               6: inverse_matrix
              }

    #Clear menu and prompt
    for n in range(0,len(options)+2):
        sys.stdout.write(CURSOR_UP)
        sys.stdout.write(ERASE_LINE)
    print("")

    #Run menu option
    if menuOption in range(1,len(options)+1):
        options[menuOption]()
    else:
        sys.exit()

    #Reload menu
    print(""),print("")
    main_menu()
    return

def basic_Matrix():
    print("--- VECTORS, MATRICES, and TENSORS ---")
    vector1=array([[1],[2],[3]])
    matrix1=array([[1,2,3],[4,5,6],[7,8,9]])
    tensor0=array([1])
    tensor1=array([1,2,3])
    tensor2=array([[1,2,3],[4,5,6],[7,8,9]])

    print("1x3 Vector:"),print(vector1),print("")
    print("3x3 Matrix:"),print(matrix1),print("")
    print("Scalar(0-D) Tensor"),print(tensor0),print("")
    print("Vector(1-D) Tensor"),print(tensor1),print("")
    print("Matrix(2-D) Tensor"),print(tensor2),print("")
    print("NOTE: Tensors can be any number of dimensions")
    return

def matrix_transpose():
    print("---TRANSPOSE A MATRIX---")
    matrixt=matrix33.T

    print("Original Matrix:"),print(matrix33),print("")
    print("Transposed Matrix:"),print(matrixt)
    return

def matrix_addition():
    print("---MATRIX ADDITION---")
    matrixsum=add(matrix33,matrix33T)
    print("Original Matrices:")
    print(matrix33),print(""),print(matrix33T),print("")
    print("Sum Matrix:"),print(matrixsum)
    return

def matrix_scalar_addition():
    print("---MATRIX/SCALAR ADDITION---")
    scalar1=array([3])
    matrixsum=add(matrix33,scalar1)
    print("Original Matrix:"),print(matrix33),print("")
    print("Scalar:"),print(scalar1),print("")
    print("Sum Matrix:"),print(matrixsum)
    return

def matrix_multiplication():
    print("---MATRIX MULTIPLICATION---")
    matrix1=array([[1,2,3],[4,5,6]])
    matrix2=array([[9,8],[6,5],[3,2]])
    ABproduct=dot(matrix1,matrix2)
    BAproduct=dot(matrix2,matrix1)

    print("Original Matrix:"),print(matrix1),print(""),print(matrix2)
    print("")
    print("Matrix A*B:"),print(ABproduct),print("")
    print("Matrix B*A:"),print(BAproduct)
    return

def inverse_matrix():
    print("---INVERSE MATRIX---")
    matrix1=matrix([[3,2,1],[4,3,2],[4,4,3]])
    imatrix=matrix1.I

    print("Original Matrix:"),print(matrix1),print("")
    print("Inverse Matrix:"),print(imatrix),print()
    print("Matrix * Inverse:"),print(dot(matrix1,imatrix))
    return

main_menu()
