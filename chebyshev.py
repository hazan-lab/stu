"""
This script compares two ways of generating monic Chebyshev polynomials of the first kind, T_n(x):

1) Windsor's recurrence-based method
    - Stores coefficients in descending powers.
    - Runtime: each iter (2 to n) does work proportional to current degree, i.e. 2, 3, ..., n ~ O(n^2)
    - Space: O(n)

2) Isabel's derived binomial summation formula
    - Directly encodes the closed-form expression and stores coefficients in ascending powers.
    - Runtime: Loop over k (roughly O(n) iters) and run an inner-loop (O(n) iters) for each k, so ~ O(n^2)
    - Space: O(n)

"""

import torch

def get_windsor_monic_chebyshev_coeffs(n: int) -> torch.Tensor:
    """
    Generate the n-th Chebyshev polynomial of the first kind T_n(x),
    in monic form (leading coefficient = 1), using an integer recurrence.

    The coefficients are returned in descending order:
        [a_n, a_{n-1}, …, a_0]
    where T_n(x) = a_n x^n + a_{n-1} x^{n-1} + … + a_0.

    Args:
        n: The polynomial degree.

    Returns:
        A 1D torch.Tensor of length (n+1) holding Tₙ(x) coefficients in descending order.
    """

    def chebyshev_t_int(n: int) -> list[int]:
        """
        Compute Chebyshev T_n(x) coefficients using the standard recurrence:
            T_0(x) = 1
            T_1(x) = x
            T_k(x) = 2x \cdot T_{k-1}(x) - T_{k-2}(x)
        
        This returns the coefficients in descending order as integers
        (unscaled, so T_n's leading coefficient is 2^{n-1} for n≥1).
        """
        if n == 0:
            return [1]
        elif n == 1:
            return [1, 0]

        T0 = [1]         # T_0(x) = 1
        T1 = [1, 0]      # T_1(x) = x
        for _ in range(2, n + 1):
            # 2x \cdot T_{k-1}(x)
            T2 = [2 * c for c in T1] + [0]
            
            # Subtract T_{k-2}(x)
            d = len(T2) - len(T0)
            padded_T0 = [0] * d + T0
            T2 = [a - b for (a, b) in zip(T2, padded_T0, strict=True)]
            
            T0, T1 = T1, T2
        
        return T2

    # Get the integer-based coefficients in descending order.
    coeffs = torch.tensor(chebyshev_t_int(n), dtype=torch.float64)

    # Scale them so that the leading term is 1 (monic). 
    # For n > 0, original leading coefficient is 2^(n-1). 
    if n > 0:
        coeffs /= (2.0 ** (n - 1))

    return coeffs


def get_isabel_monic_chebyshev_coeffs(n: int) -> list[float]:
    """
    Implement Isabel's binomial summation for T_n(x):

        T_n(x) = sum_{k=0}^{floor(n/2)} sum_{m=k}^{floor(n/2)} 
                  [ C(n, 2m) \cdot C(m, k) \cdot (-1)^k ] \cdot x^(n - 2k)

    where C(a, b) = "a choose b". After computing this polynomial, we scale 
    it so the leading coefficient equals 1 (making it monic).

    The resulting coefficients are returned in ascending order:
        [a_0, a_1, …, a_n]
    so T_n(x) = a_0 + a_1 x + … + a_n x^n.

    Args:
        n: The polynomial degree.

    Returns:
        A list of length (n+1) holding T_n(x) coefficients in ascending order.
    """
    from math import comb
    
    max_m = n // 2

    # We'll store the coefficient for x^i at index i, i = 0..n
    coeffs = [0.0] * (n + 1)

    # Core double sum:
    #    sum_{k=0}^{floor(n/2)} sum_{m=k}^{floor(n/2)} [C(n,2m) C(m,k) (-1)^k] * x^(n - 2k)
    for k in range(max_m + 1):
        partial_sum = 0
        for m in range(k, max_m + 1):
            partial_sum += comb(n, 2*m) * comb(m, k) * ((-1) ** k)
        
        exponent = n - 2*k
        coeffs[exponent] = partial_sum

    # Scale so the leading term (coefficient of x^n) is 1, if n>0.
    if n > 0:
        lead = coeffs[n]
        coeffs = [c / lead for c in coeffs]

    return coeffs


def compare_chebyshev_coeffs(num_vals=range(9)):
    """
    Compare the two Chebyshev polynomial computations for a
    range of degrees n in num_vals. Prints the coefficients in ascending
    order for both implementations, along with their absolute difference.
    """
    for n in num_vals:
        # 1) Windsor's version is in descending order (Isabel's is in ascending order)
        windsor_descending = get_windsor_monic_chebyshev_coeffs(n).tolist()
        windsor_ascending = windsor_descending[::-1]

        # 3) Isabel’s version
        isabel_ascending = get_isabel_monic_chebyshev_coeffs(n)

        # 4) Compare
        diffs = [abs(a - b) for a, b in zip(windsor_ascending, isabel_ascending, strict=True)]

        print(f"n={n}:")
        print("  Windsor (ascending) =", [round(c, 6) for c in windsor_ascending])
        print("  Isabel  (ascending) =", [round(c, 6) for c in isabel_ascending])
        print("  Diffs               =", [round(d, 6) for d in diffs])
        print()


if __name__ == "__main__":
    # Compare degrees n=0..9
    compare_chebyshev_coeffs(range(9))
