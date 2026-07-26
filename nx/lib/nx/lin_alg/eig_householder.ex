defmodule Nx.LinAlg.EigHouseholder do
  @moduledoc """
  Householder reflector generation and application.

  Ported from LAPACK DLARFG (generate) and DLARF1F (apply).
  """

  @doc """
  DLARFG: Generate an elementary reflector (Householder).

  H = I - tau * v * v', where H * x = [beta, 0, ..., 0]'.
  v is normalized so that v[0] = 1.

  Returns {v, tau, beta}.
  """
  def dlarfg(x) do
    n = Nx.size(x)

    if n == 1 do
      {Nx.tensor([1.0], type: :f64), 0.0, Nx.to_number(Nx.reshape(x[0..0], {}))}
    else
      alpha = Nx.to_number(Nx.reshape(x[0..0], {}))
      xnorm = Nx.to_number(Nx.LinAlg.norm(x[1..-1//1]))

      if xnorm == 0.0 do
        {Nx.concatenate([Nx.tensor([1.0]), Nx.broadcast(0.0, {n - 1})]), 0.0, alpha}
      else
        # LAPACK DLARFG:
        # beta = -sign(alpha) * ||x||
        norm = :math.sqrt(alpha * alpha + xnorm * xnorm)
        beta = if alpha > 0, do: -norm, else: norm

        # v = [1, x(2:)/(alpha - beta)]
        # alpha - beta = alpha + sign(alpha)*norm
        denom = alpha - beta
        v_tail = Nx.divide(x[1..-1//1], denom)
        v = Nx.concatenate([Nx.tensor([1.0]), v_tail])

        # tau = (beta - alpha) / beta
        tau = if beta == 0.0, do: 0.0, else: (beta - alpha) / beta

        {v, tau, beta}
      end
    end
  end

  @doc """
  DLARF: Apply an elementary reflector H = I - tau * v * v' to a matrix.

  side = :left:   H * A
  side = :right:  A * H
  """
  def dlarf(v, tau, a, side \\ :left)

  def dlarf(v, tau, a, :left) do
    m = Nx.size(v)
    n = div(Nx.size(a), m)
    a2 = Nx.reshape(a, {m, n})

    # H * A = A - tau * v * (v' * A)
    vt = Nx.new_axis(v, 0)
    v_c = Nx.new_axis(v, 1)
    vta = Nx.dot(vt, a2)
    correction = Nx.dot(v_c, Nx.multiply(tau, vta))
    Nx.subtract(a2, correction)
  end

  def dlarf(v, tau, a, :right) do
    n = Nx.size(v)
    m = div(Nx.size(a), n)
    a2 = Nx.reshape(a, {m, n})

    # A * H = A - tau * (A * v) * v'
    v_c = Nx.new_axis(v, 1)
    vt = Nx.new_axis(v, 0)
    av = Nx.dot(a2, v_c)
    correction = Nx.dot(Nx.multiply(tau, av), vt)
    Nx.subtract(a2, correction)
  end
end
