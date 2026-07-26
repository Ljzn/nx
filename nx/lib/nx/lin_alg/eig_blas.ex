defmodule Nx.LinAlg.EigBlas do
  @moduledoc """
  BLAS wrappers and DLARTG for Eigenvalue Decomposition.

  Provides LAPACK-style BLAS operations using Nx primitives.
  """

  @doc """
  DAXPY: y = a * x + y (element-wise vector operation).
  """
  def daxpy(a, x, y) do
    Nx.add(Nx.multiply(a, x), y)
  end

  @doc """
  DCOPY: copy vector x to y.
  """
  def dcopy(x) do
    x
  end

  @doc """
  DSCAL: x = a * x
  """
  def dscal(a, x) do
    Nx.multiply(a, x)
  end

  @doc """
  DSWAP: swap vectors x and y.
  Returns {y, x}.
  """
  def dswap(x, y) do
    {y, x}
  end

  @doc """
  DROT: apply Givens rotation to (x, y).
    x = c * x + s * y
    y = -s * x + c * y  (using new x)
  Note: this is a simplified version; LAPACK's DROT applies element-wise.
  """
  def drot(x, y, c, s) do
    x_new = Nx.add(Nx.multiply(c, x), Nx.multiply(s, y))
    y_new = Nx.subtract(Nx.multiply(-s, x), Nx.multiply(c, y))
    {x_new, y_new}
  end

  @doc """
  DNRM2: Euclidean norm (2-norm) of vector x.
  """
  def dnrm2(x) do
    Nx.to_number(Nx.LinAlg.norm(x))
  end

  @doc """
  IDAMAX: index (1-based) of first element having maximum absolute value.
  """
  def idamax(x) do
    idx = Nx.argmax(Nx.abs(x))
    # LAPACK uses 1-based indexing, but we return 0-based for Elixir
    Nx.to_number(idx) + 1
  end

  @doc """
  DNRM2 variant: sqrt(|x₁|² + |x₂|² + ... + |xₙ|²)
  """
  def dnrm2_value(x) do
    :math.sqrt(Nx.to_number(Nx.sum(Nx.multiply(x, x))))
  end

  @doc """
  DLARTG: generate Givens rotation (LAPACK).

  Given f and g, returns {c, s, r} such that:
    [c s; -conj(s) c]^T * [f; g] = [r; 0]
  where c and s are cosine and sine of the rotation angle.

  Ported from LAPACK DLARTG.
  """
  def dlartg(f, g) do
    r = Nx.LinAlg.EigUtil.dlapy2(f, g)

    if r == 0.0 do
      {1.0, 0.0, 0.0}
    else
      c = f / r
      s = g / r
      {c, s, r}
    end
  end

  @doc """
  DLARTG with overflow protection (LAPACK exact algorithm).
  More numerically robust for extreme values.
  """
  def dlartg_safe(f, g) do
    abs_g = abs(g)
    safmin = Nx.LinAlg.EigUtil.dlamch("S")

    if abs_g < safmin do
      # Use LAPACK's DLARTGP fallback for tiny g
      dlartgp(f, g)
    else
      r = Nx.LinAlg.EigUtil.dlapy2(f, g)
      c = f / r
      s = g / r
      {c, s, r}
    end
  end

  # LAPACK DLARTGP: Givens rotation for tiny g values.
  defp dlartgp(f, g) do
    abs_f = abs(f)
    abs_g = abs(g)

    cond do
      abs_f == 0.0 and abs_g == 0.0 ->
        {1.0, 0.0, 0.0}

      abs_f >= abs_g ->
        ratio = abs_g / abs_f
        rt = :math.sqrt(1.0 + ratio * ratio)
        r = abs_f * rt
        cs = 1.0 / rt
        sn = ratio / rt
        c = if f > 0.0, do: cs, else: -cs
        s = if g > 0.0, do: sn, else: -sn
        {c, s, r}

      true ->
        ratio = abs_f / abs_g
        rt = :math.sqrt(1.0 + ratio * ratio)
        r = abs_g * rt
        sn = 1.0 / rt
        cs = ratio / rt
        c = if f > 0.0, do: cs, else: -cs
        s = if g > 0.0, do: sn, else: -sn
        {c, s, r}
    end
  end
end
