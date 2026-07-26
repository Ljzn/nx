defmodule Nx.LinAlg.EigUtil do
  @moduledoc """
  Utility functions for Eigenvalue Decomposition.

  Ported from LAPACK auxiliary routines (DLAMCH, ILAENV, LSAME,
  XERBLA, DISNAN, DLAPY2, DLANGE).
  """

  @doc """
  Machine constants (LAPACK DLAMCH).
  Returns double-precision machine parameters.
  """
  def dlamch(cmach) do
    # IEEE 754 double precision constants
    case String.upcase(cmach) do
      # eps: relative machine precision
      "E" -> 2.2204460492503131e-16
      # sfmin: safe minimum (tiny)
      "S" -> 2.2250738585072014e-308
      # base: radix
      "B" -> 2.0
      # prec: eps * base
      "P" -> 2.2204460492503131e-16
      # t: mantissa digits
      "N" -> 53
      # rnd: rounding mode (1=?) 
      "R" -> 1.0
      # emin: minimum exponent
      "M" -> -1021
      # underflow threshold
      "U" -> 2.2250738585072014e-308
      # emin - 1 + t
      "L" -> -1021 + 53 - 1
      # overflow threshold
      "O" -> 1.7976931348623157e+308
      _ -> raise ArgumentError, "dlamch: unknown cmach=#{inspect(cmach)}"
    end
  end

  @doc """
  Block size query (LAPACK ILAENV).
  Returns recommended block sizes for blocked algorithms.
  """
  def ilaenv(ispec, _name \\ "", _opts \\ "", _n1 \\ 0, _n2 \\ 0, _n3 \\ 0, _n4 \\ 0)

  # NB: optimal block size
  def ilaenv(1, _, _, _, _, _, _), do: 64
  # NBMIN: minimum block size
  def ilaenv(2, _, _, _, _, _, _), do: 2
  # NX: crossover point
  def ilaenv(3, _, _, _, _, _, _), do: 32
  # NXB: minimum block size for blocking
  def ilaenv(4, _, _, _, _, _, _), do: 64
  # NBIT: number of bits in integer
  def ilaenv(5, _, _, _, _, _, _), do: 32
  # NBMIN for DORMQR
  def ilaenv(6, _, _, _, _, _, _), do: 2
  # NBMIN for DORMLQ
  def ilaenv(7, _, _, _, _, _, _), do: 2
  # NB for DGEQRF
  def ilaenv(8, _, _, _, _, _, _), do: 64
  # NB for DORMQR
  def ilaenv(9, _, _, _, _, _, _), do: 64
  # NB for DORMLQ
  def ilaenv(10, _, _, _, _, _, _), do: 64
  # NB for DGEBRD
  def ilaenv(11, _, _, _, _, _, _), do: 64
  # NMIN: crossover for DHSEQR
  def ilaenv(12, _, _, _, _, _, _), do: 75
  # KCL: clustering threshold
  def ilaenv(13, _, _, _, _, _, _), do: 1
  # NIBBLE: nibble interval
  def ilaenv(14, _, _, _, _, _, _), do: 3
  # MAXB: max block size
  def ilaenv(15, _, _, _, _, _, _), do: 128
  # NBMIN for DLARFB
  def ilaenv(16, _, _, _, _, _, _), do: 3
  # NB for multi-shift QR
  def ilaenv(17, _, _, _, _, _, _), do: 64
  # NSWEEP: number of sweeps (unused)
  def ilaenv(18, _, _, _, _, _, _), do: 3

  @doc """
  Error handler (LAPACK XERBLA). Raises an ArgumentError.
  """
  def xerbla(name, info) do
    raise ArgumentError, "error in #{inspect(name)}, info=#{info}"
  end

  @doc """
  Character comparison (LAPACK LSAME). Case-insensitive.
  """
  def lsame(ca, cb) do
    String.downcase(ca) == String.downcase(cb)
  end

  @doc """
  NaN test (LAPACK DISNAN). Returns true if the value is NaN.
  """
  def disnan(din) do
    is_float(din) and :math.log(din) != :math.log(din)
    # Alternative: din != din (NaN is the only value where NaN != NaN)
  end

  @doc """
  Sqrt(x² + y²) without overflow (LAPACK DLAPY2).
  """
  def dlapy2(x, y) do
    abs_x = abs(x)
    abs_y = abs(y)

    if abs_x > abs_y do
      ratio = abs_y / abs_x
      abs_x * :math.sqrt(1.0 + ratio * ratio)
    else
      if abs_y == 0.0 do
        0.0
      else
        ratio = abs_x / abs_y
        abs_y * :math.sqrt(1.0 + ratio * ratio)
      end
    end
  end

  @doc """
  Matrix 1-norm (LAPACK DLANGE with norm='1' or 'O').
  Returns max column sum of absolute values.
  """
  def dlange_one(a) do
    # ||A||_1 = max_j sum_i |a_ij|
    col_sums = Nx.sum(Nx.abs(a), axes: [0])
    Nx.to_number(Nx.reduce_max(col_sums))
  end

  @doc """
  Matrix infinity-norm (LAPACK DLANGE with norm='I').
  Returns max row sum of absolute values.
  """
  def dlange_inf(a) do
    # ||A||_inf = max_i sum_j |a_ij|
    row_sums = Nx.sum(Nx.abs(a), axes: [1])
    Nx.to_number(Nx.reduce_max(row_sums))
  end

  @doc """
  Matrix Frobenius norm (LAPACK DLANGE with norm='F').
  """
  def dlange_fro(a) do
    Nx.to_number(Nx.sqrt(Nx.sum(Nx.pow(Nx.abs(a), 2))))
  end
end
