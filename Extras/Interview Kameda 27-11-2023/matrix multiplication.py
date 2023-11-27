
#functions
def print_matrix(matrix):
    for row in matrix:
        print(row)

def multiplyhadmad(matrix_a, matrix_b):
    length = len(matrix_a)
    result =[]
    for i in range(length):
        row = []
        for j in range(length):
            row.append(matrix_a[i][j] * matrix_b[i][j])
        result.append(row)
    return result

def multiply(matrix_a, matrix_b):
    length = len(matrix_a)
    result =[]
    for i in range(length):
        row = []
        for j in range(length):
            ele = 0
            for k in range(length):
                ele += matrix_a[i][k] * matrix_b[k][j]
            row.append(ele)
        result.append(row)
        
    return result

#Matrix
matrix_a = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

matrix_b = [
    [7,8,9],
    [4,5,6],
    [1,2,3]
]

#function call
hadamad_product = multiplyhadmad(matrix_a, matrix_b)
matrixproduct = multiply(matrix_a, matrix_b)

#output
print("\nMatrix A = ")
print_matrix(matrix_a)
print("\nMatrix B = ")
print_matrix(matrix_b)
print("\nHadamad roduct =")
print_matrix(hadamad_product)
print("\nMatrix Product A*B =")
print_matrix(matrixproduct)