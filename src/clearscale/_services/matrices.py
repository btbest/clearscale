from typing import Any, Mapping, Tuple

FloatVector = Tuple[float, ...]
FloatMatrix = Tuple[FloatVector, ...]

DETERMINANT_SINGULARITY_TOLERANCE = 1e-12


def matrix_shape(matrix: FloatMatrix) -> Tuple[int, int]:
    return len(matrix), len(matrix[0])


def matrix_transpose(matrix: FloatMatrix) -> FloatMatrix:
    return tuple(tuple(row[i] for row in matrix) for i in range(len(matrix[0])))


def matrix_multiply(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
    if len(left[0]) != len(right):
        raise ValueError(f"Cannot multiply matrices with shapes {matrix_shape(left)} and {matrix_shape(right)}")
    right_t = matrix_transpose(right)
    return tuple(tuple(sum(a * b for a, b in zip(row, col)) for col in right_t) for row in left)


def matrix_vector_multiply(matrix: FloatMatrix, vector: FloatVector) -> FloatVector:
    if len(matrix[0]) != len(vector):
        raise ValueError(f"Cannot multiply {matrix_shape(matrix)} matrix by vector of length {len(vector)}")
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def is_identity_matrix(matrix: FloatMatrix, *, tolerance: float = DETERMINANT_SINGULARITY_TOLERANCE) -> bool:
    rows, cols = matrix_shape(matrix)
    if rows != cols:
        return False

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            expected = 1.0 if i == j else 0.0
            if abs(value - expected) > tolerance:
                return False

    return True


def is_diagonal_matrix(matrix: FloatMatrix, *, tolerance: float = DETERMINANT_SINGULARITY_TOLERANCE) -> bool:
    return all(abs(value) < tolerance for i, row in enumerate(matrix) for j, value in enumerate(row) if i != j)


def is_rotation_matrix(rotation: FloatMatrix, *, tolerance: float = DETERMINANT_SINGULARITY_TOLERANCE) -> bool:
    determinant = matrix_determinant(rotation)
    if abs(determinant - 1.0) > tolerance:
        return False
    for i, row in enumerate(rotation):
        norm = sum(v * v for v in row)
        if abs(norm - 1.0) > tolerance:
            return False
        for j, other_row in enumerate(rotation[i + 1 :], start=i + 1):
            dot = sum(x * y for x, y in zip(row, other_row))
            if abs(dot) > tolerance:
                return False
    # No need to check cols: For square matrices, all rows normal + orthogonal also implies orthonormal columns.
    return True


def matrix_determinant(matrix: FloatMatrix) -> float:
    """
    Calculate for arbitrary matrices using Gaussian elimination with partial pivoting.

    This could be substantially simpler for 1x1, 2x2 and 3x3 cases, but 4x4 and 5x5 will probably
    be commonplace too. Might as well solve generally, still unlikely to be a performance bottleneck.

    This simplified implementation is mostly equivalent to mpmath's `det(matrix)`, at
    https://github.com/mpmath/mpmath/blob/6d356df80ffa78a8cb14066536c4879fdfd2f344/mpmath/matrices/linalg.py#L572
    """
    rows, cols = matrix_shape(matrix)
    if rows != cols:
        raise ValueError(f"Cannot calculate determinant of non-square matrix with shape {(rows, cols)}")
    work = [list(row) for row in matrix]
    determinant = 1.0
    for col in range(rows):
        # Partial pivoting: Pivot is the row with the largest absolute value in this column
        pivot_row = max(range(col, rows), key=lambda row: abs(work[row][col]))
        if abs(work[pivot_row][col]) <= DETERMINANT_SINGULARITY_TOLERANCE:
            return 0.0  # Max across this col is 0 = entire col is 0 = singular
        if pivot_row != col:
            work[col], work[pivot_row] = work[pivot_row], work[col]
            determinant *= -1.0
        pivot_value = work[col][col]  # Swap ensures the pivot is now on the diag
        determinant *= pivot_value  # Accumulate the diagonal as we go
        for row in range(col + 1, rows):  # Subtract out pivot row from lower rows
            factor = work[row][col] / pivot_value
            for k in range(col, rows):
                work[row][k] -= factor * work[col][k]
    return determinant  # `work` is now upper-triangular; det has already been accumulated


def matrix_invert(matrix: FloatMatrix) -> FloatMatrix:
    """
    Calculate using a Gaussian elimination process.
    First expand MxM into 2MxM by appending an MxM identity matrix on the right side.
    Then swap over, turning the original left side into identity and the right side into the inverse.
    """
    rows, cols = matrix_shape(matrix)
    if rows != cols:
        raise ValueError(f"Cannot invert non-square matrix with shape {(rows, cols)}")
    # Append identity on the right
    work = [list(row) + [1.0 if i == j else 0.0 for j in range(rows)] for i, row in enumerate(matrix)]
    for col in range(rows):
        # Partial pivoting: Pivot is the row with the largest absolute value in this column
        pivot = max(range(col, rows), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) <= DETERMINANT_SINGULARITY_TOLERANCE:
            # Max across this col is 0 = entire col is 0 = singular
            raise ValueError("Matrix is singular and cannot be inverted.")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]  # Pivot now on the diagonal
        work[col] = [v / pivot_value for v in work[col]]  # Normalize by pivot to make it 1.0
        for row in range(rows):  # Eliminate pivot above and below current row
            if row == col:
                continue
            factor = work[row][col]  # No normalization here; pivot row (`col`) is already normalized
            work[row] = [val - factor * piv for val, piv in zip(work[row], work[col])]
    return tuple(tuple(row[rows:]) for row in work)  # Return right half; which is now the inverse
