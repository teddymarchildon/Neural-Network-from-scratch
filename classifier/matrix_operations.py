"""
Defines a set of functions to be used on various math structures
"""

def dot_product(vector1, vector2):
	"""
	:param vector1: the first matrix in the dot product
	:param vector2: the second matrix in the dot product, although of course it does not matter since the dot product is commutative
	:return: the scalar value of the dot product
	"""

	# the dot product is defined for vectors, or matrices with one row/column
	if len(vector1) == 0 or len(vector2) == 0:
			raise ValueError("Inputs should have values")
	
	sum = 0
	for i in range(0, len(vector1)):
		sum += (vector1[i] * vector2[i])
	
	return sum

def scalar_product(scalar, matrix):
	"""
	:input scalar: the factor to multiply the matrix with
	:input matrix: the matrix to be scaled
	:return: a new matrix scaled by the factor
	"""
	if len(matrix) == 0:
		raise ValueError("Input matrix should have length")
	
	num_rows = len(matrix)
	num_cols = len(matrix[0])
	result = [[]]
	for i in range(0, num_rows):
		for j in range(0, num_cols):
			result[i][j] = matrix[i][j] * scalar
	
	return result

def add_matrix(matrix1, matrix2):
	"""
	:param matrix1: the left matrix in the addition
	:param matrix2: the right matrix in the addition, although of course order does not matter
	:return: a new matrix with the addition performed
	"""
	if len(matrix1) == 0 or len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
		raise ValueError("Input matrices should be of the same dimension")
	
	num_rows = len(matrix1)
	num_cols = len(matrix1[0])
	result = [[]]
	for i in range(0, num_rows):
		for j in range(0, num_cols):
			result[i][j] = matrix1[i][j] + matrix2[i][j]

	return result

def multiply_matrix(matrix1, matrix2):
	"""
	:param matrix1: the left matrix in the multiplication
	:param matrix2: the right matrix in the multiplication
	:return: a new matrix with the result
	"""

	if len(matrix1) == 0 or len(matrix2) == 0:
		raise ValueError("Input matrices should be of the same dimension")
	
	# number of columns in the left matrix must be equal to the number of rows in right matrix
	if len(matrix1[0]) != len(matrix2):
		raise ValueError("Left matrix should have as many columns as the right matrix has rows")
		
	num_rows = len(matrix2)
	num_cols = len(matrix1[0])
	result = [[]]
	for i in range(num_rows):
		current_entry = 0
		for j in range(num_cols):
			left = matrix1[i][j]
			right = matrix2[j][i]
			current_entry += (left * right)
		result[i][j] = current_entry
	
	return result
