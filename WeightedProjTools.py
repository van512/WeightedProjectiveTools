import numpy as np
from sympy.solvers.diophantine import diophantine
from sympy import symbols, Eq, solve, parse_expr
from sympy.utilities.lambdify import lambdify


"""Provides some tools and functions to perform calculations in the context of weighted projective space."""


###############
def dimrec(a: list, d: int, i=0, l=0):
    """
    Recursive function to calculate dim C_a[X]_d, i.e. count combinations of numbers from list 'a' that sum up to 'd'.
    
    Parameters:
    a (list): A list of integers : the weights a=(a_0,\dots,a_n).
    d (int): Target sum to achieve : the degree of the weighted homogeneous polynomials.
    i (int): Current index in the list (default is 0).
    l (int): Current accumulated sum (default is 0).
    
    Returns:
    int: Number of ways to combine elements from 'a' to sum up to 'd', equal to dim C_a[X]_d
    """
    
    # Initialize the current count of valid combinations
    current_sum = 0
    
    # If we haven't reached the end of the list
    if i < len(a):
        # Loop over all possible values of the current element from 0 to d-l, stepping by a[i]
        for ki in range(0, d - l + 1, a[i]):
            # Recursively check combinations using the next element
            current_sum += dimrec(a, d - l, i + 1, ki)
    
    # Base case: if we used all elements and the remaining sum is exactly 0
    elif d - l == 0:
        assert i == len(a)  # Ensure we've processed all elements
        current_sum = 1  # This counts as one valid combination
    
    # Return the total number of valid combinations found
    return current_sum


###############
def uniquebi(a:int, s:int, d:int, verbose=False):
    """Solves the Diophantine equation d = b_i(d)a_i+c_i(d)s_i with the notation of the thesis
    Input Variables :
    a=a_i, s=s_i, d=degree
    Returns :
    The unique b=b_i(d) such that 0<=b<s
    """
    # Define symbols for b, c, x integers
    b, c, x = symbols("b, c, x", integer=True)
    syms = (b, c)

    # Define the Diophantine equation to solve: a*b + s*c = d
    equ = a * b + s * c - d
    
    # Solve the equation for b and c, getting a solution with a symbolic parameter
    solutions, = diophantine(equ, syms=(b, c))

    # Extract and define the symbolic parameter (usually t_0) from the solution
    parameters = set().union(*(s.free_symbols for s in solutions))
    param_t = symbols(str(list(parameters)[0]))

    # Extract expressions for b and c from the solution tuple
    expr_b = str(list(solutions)[0])
    expr_c = str(list(solutions)[1])

    # Parse b and c expressions into usable SymPy expressions
    parsed_b = parse_expr(expr_b, transformations="all")
    parsed_c = parse_expr(expr_c, transformations="all")

    # Solve the expression for b in terms of the parameter t
    sol_t = solve(parsed_b, param_t)[0]
    
    # Create a lambda function to evaluate t using NumPy
    lambd_t = lambdify(x, sol_t * x, 'numpy')
    t_val = int(lambd_t(1))  # Get the integer value of t

    # Substitute t_val into the expression for b
    b_val = parsed_b.subs({param_t: t_val})
    
    # Adjust t_val to ensure 0 <= b < s by incrementing/decrementing t
    i = 0
    while b_val >= s or b_val < 0:
        t_val = t_val - 1 * (i - 1) + 2 * i  # Alternate decrement and increment
        b_val = parsed_b.subs({param_t: t_val})
        i += 1

    # Calculate the corresponding c value
    c_val = parsed_c.subs({param_t: t_val})

    # Debug prints 
    if verbose == True:
      print(f"{d}={b_val}*{a}+{c_val}*{s}={b_val*a+c_val*s}")
      print(b_val < s)
      print(b_val >= 0)

    # Return the valid b value
    return b_val


##################
def reduce_arr(func, arr: list):
    """
    Applies a reduction function to the array with each element individually removed.
    
    Parameters:
    func: A function with a 'reduce' method (e.g., numpy's ufuncs like np.gcd, np.lcm).
    arr (list): Input list of numbers.
    
    Returns:
    output: A list where each entry is the result of applying 'func' to the array with one element left out.
    """
    
    output = []
    
    # Loop through each index of the array
    for i in range(len(arr)):
        # Create a new array with the i-th element removed
        arrhat = np.concatenate((arr[:i], arr[i+1:])).astype(int)
        
        # Apply the reduction function to the modified array and store the result
        output.append(func.reduce(arrhat))
    
    # Return the list of results
    return output


#########
class Weights:
    """
    Represents a list of weights and computes its reduced and well-formed equivalents
    """
    def __init__(self, weights: list):
        weights.sort() # sort the weights 
        self.weights = weights
        # Compute the greatest common divisor (GCD) of all weights
        self.pgcd = np.gcd.reduce(weights)
        # Compute the reduced weights by dividing each weight by the GCD
        self.reduced_weights = (weights / self.pgcd).astype(int)
        # Compute GCDs of the reduced weights with each element removed
        self.S = reduce_arr(np.gcd, self.reduced_weights)
        # Compute the least common multiples (LCM) of these GCDs
        self.Q = reduce_arr(np.lcm, self.S)
        # Calculate the well-formed weights
        self.wellformed_weights = (np.array(self.reduced_weights) / self.Q).astype(int)
        # Compute the LCM of the GCD sublist S
        self.q = np.lcm.reduce(self.S)


class SheafIsomorphism:
    """
    Represents a sheaf isomorphism with dimension reduction using weights.
    """
    def __init__(self, W: Weights, degree: int):
        self.W = W  # Associated weight class
        self.degree = degree  # Original degree before weight reduction

        # Compute the dimension before reduction
        self.dim_before = dimrec(self.W.weights, self.degree)

        # Calculate the well-formed degree if possible
        self.form_well_degree()


    def form_well_degree(self):
        """
        Computes the well-formed degree if the degree can be reduced.
        """
        temp = self.degree / self.W.pgcd
        precision = 1e-10
        
        # Check if the reduced degree is effectively an integer (handles floating-point precision)
        if abs(temp - int(temp)) < precision:
            # 
            self.isreducible = True

            # Set the reduced degree
            self.reduced_degree = int(temp)
            
            # Calculate unique bi values for each (ai, si) pair
            self.B = [uniquebi(ai, si, self.reduced_degree) for ai, si in zip(self.W.reduced_weights, self.W.S)]
            
            # Compute the well-formed degree using the formula: phi(d) in the thesis
            self.wellformed_degree = (self.reduced_degree - np.dot(self.B, self.W.reduced_weights)) / self.W.q
            
            # Compute the dimension after reduction
            self.dim_after = dimrec(self.W.wellformed_weights, self.wellformed_degree)
        else:
            self.isreducible = False


    def dimensions_agree(self):
        """
        Checks if the dimensions before and after reduction agree.
        """
        print(self.dim_before, self.dim_after)
        return self.dim_before == self.dim_after
