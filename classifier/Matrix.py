
class Matrix (object):
    """
    The representation for the matrix
    """

    def __init__(self, matrix):
        """
        :param input: the input parameter. A list of lists describing the matrix is expected here
        Initializes the matrix class
        """
        if len(matrix) == 0:
            raise ValueError("Ensure the input has rows")

        self.num_rows = len(matrix)
        self.num_cols = len(matrix[0])
        self.object = matrix
    
    def get_object(self):
        """
        :return: the matrix object
        """
        return self.object
    
    def get_rows(self):
        """
        :return: the rows of the matrix
        """
        return [row for row in self.object]
    
    def get_cols(self):
        """
        :return: the columns of the matrix
        """
        cols = []
        for col in range(0, self.num_cols):
            current_column = []
            for row in range(0, self.num_rows):
                current_column.append(self.object[row][col])
            cols.append(current_column)
        return cols
                

if __name__ == "__main__":
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    cols = m.get_rows()
    print(cols)
