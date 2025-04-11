import numpy as np # type: ignore
from sympy.solvers.diophantine import diophantine # type: ignore
from sympy import symbols, solve, parse_expr # type: ignore
from sympy.utilities.lambdify import lambdify # type: ignore
from functools import lru_cache
from itertools import combinations
from numpy.typing import NDArray # type: ignore
from scipy.special import comb # type: ignore
#import time


"""Provides some tools and functions to perform calculations in the context of weighted projective space."""


def dimrec(a: list, d: int) -> int:
    """
    Optimized recursive function with memoization to calculate dim C_a[X]_d.
    """

    @lru_cache(maxsize=None)
    def helper(i: int, remaining: int) -> int:
        if i == len(a):
            return 1 if remaining == 0 else 0
        
        total = 0
        # Try all multiples of a[i] that do not exceed 'remaining'
        for k in range(0, remaining + 1, a[i]):
            total += helper(i + 1, remaining - k)
        
        return total

    return helper(0, d)


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


def reduce_arr(func, arr: list) -> list:
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


class Weights:
    """
    Represents a list of weights and computes its reduced and well-formed equivalents
    """
    def __init__(self, weights: list):
        weights.sort() # sort the weights 
        self.weights = np.array(weights)
        # Compute the greatest common divisor (GCD) of all weights
        self.pgcd = np.array(np.gcd.reduce(weights))
        # Compute the reduced weights by dividing each weight by the GCD
        self.reduced_weights = np.array((weights / self.pgcd).astype(int))
        # Compute GCDs of the reduced weights with each element removed
        self.S = np.array(reduce_arr(np.gcd, self.reduced_weights))
        # Compute the least common multiples (LCM) of these GCDs
        self.Q = np.array(reduce_arr(np.lcm, self.S))
        # Calculate the well-formed weights
        self.wellformed_weights = np.array((self.reduced_weights / self.Q).astype(int))
        # Compute the LCM of S
        self.q = np.array(np.lcm.reduce(self.S))
        # lcm of the weights
       # self.a = np.array(np.lcm.reduce(self.weights))


class LinearSystem:
    """
    Represents a linear system and reduces it when possible, calculates its dimension.
    """
    def __init__(self, W: Weights, degree: int):
        self.W = W  # Associated weight class
        self.degree = degree  # Original degree before weight reduction
        
        self.isreducible = None
        self.reduced_degree = None
        self.wellformed_degree = None

        # Calculate the well-formed degree if possible
        self.form_well_degree()

        # Computes the dimension on the normalized weights and degree (normalized = reduced + well-formed)
        if self.wellformed_degree is not None:
            self.dimension = dimrec(self.W.wellformed_weights, self.wellformed_degree)
        else :
            self.dimension = dimrec(self.W.weights, self.degree)

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

        else:
            self.isreducible = False

    
    #def is_ample(self):
    #    if self.wellformed_degree is not None:
    #       return self.wellformed_degree == np.array(np.lcm.reduce(self.W.wellformed_weights)) and self.wellformed_degree > 0
    #    else:
    #        return self.degree == np.array(np.lcm.reduce(self.W.weights)) and self.degree > 0

    #def is_very_ample(self):

#class TwistedSheaf(LinearSystem):
#    def __init__(self, degree: int, weights: Weights):
#        super().__init__(weights, degree)
        
    #def is_ample(self):
     #   if self.degree == np.array(np.lcm.reduce(self.W.self.weights))
    #def is_very_ample(self):
        # Check if the sheaf is very ample



####  to check very ampleness

def very_ample_bound(weights:NDArray[np.int32]):
    # G(Q) G(a) for us

        # Function to calculate LCM of a list using numpy's np.lcm.reduce
    @lru_cache(None)  # Memoize this function to avoid recomputing the same sublist
    def memoized_lcm(weights_sublist):
        return np.lcm.reduce(weights_sublist)

    def sum_lcms(nu): #nu = length of weights_sublist
        total_lcm_sum = 0
        for weights_sublist in combinations(weights, nu):
            # Sum the LCM of the elements of the sublist of weights
            total_lcm_sum += memoized_lcm(weights_sublist)
        return total_lcm_sum

    r = len(weights)-1
    if r==0:
        return -weights[0]
    elif r>0:
        temp_sum = 0
        for nu in range(2,r+1+1):
            temp_sum += sum_lcms(nu)/comb(r-1,nu-2)
        return -weights.sum() + temp_sum/r



class WeightedProjectiveSpace:
    
    def __init__(self, W:Weights):
        self.W = W # weights
        self.embedding_dimension = self.embeds_into()

    def embeds_into(self)->int:
        m = np.array(np.lcm.reduce(self.W.wellformed_weights))
        nGm = np.ceil(very_ample_bound(self.W.wellformed_weights)/m)
        G = very_ample_bound(self.W.wellformed_weights)
        self.m = m
        self.G = G
        self.nGm1 = G/m
        self.nGm = nGm
        if nGm < 1:
            deg_mn = m  #0 pb chekc >0
        else:
            deg_mn = nGm*m
        linsys = LinearSystem(Weights(self.W.wellformed_weights), np.array(np.int64(deg_mn)))
        self.embedding_linear_system = linsys
        N = linsys.dimension-1
        return N
        # return the dimension of the projective space in which it embeds into


w09 = Weights([1,4,5,10])
w10 = Weights([1,2,6,9])
w11 = Weights([1,2,3,6])
w12 = Weights([1,3,8,12])
w13 = Weights([1,6,14,21])
w14 = Weights([2,3,10,15])
w15 = Weights([1,6,10,15])

sigmas = [20,18,12,24,42,30,60] # table 3
gplusone = [22,29,25,25,22,16,444] # table 2
numbers = ['9','10','11','12','13','14','BelRob']

i =0 
for w in [w09, w10, w11, w12, w13, w14,w15]:
    wps=WeightedProjectiveSpace(w)
    print(" CASE ", numbers[i])
    print("weights Q = ", wps.W.wellformed_weights)
    print("m=lcm(Q) = ", wps.m)
    print("G(Q) = ",wps.G)
    print("G(Q)/m =",wps.nGm1)
    print("G(Q)/m < n = ", wps.nGm)
    print("deg = nm = ", wps.embedding_linear_system.degree)
    print("dimension N = ", wps.embedding_dimension)
    print("Bruno's sigma = nm = ", sigmas[i])
    print("Bruno's N = g+1 = ", gplusone[i])
    print("-------------------")
    i+=1