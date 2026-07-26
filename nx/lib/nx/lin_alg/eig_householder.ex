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
    x0 = Nx.to_number(Nx.reshape(x[0..0], {}))

    if n == 1 do
      {Nx.tensor([1.0], type: :f64), 0.0, x0}
    else
      tail = x[1..-1//1]
      sigma = Nx.sum(Nx.pow(Nx.abs(tail), 2)) |> Nx.to_number()
      alpha = x0

      if sigma == 0.0 and alpha >= 0.0 do
        {Nx.concatenate([Nx.tensor([1.0]), Nx.broadcast(0.0, {n - 1})]), 0.0, alpha}
      else
        norm = :math.sqrt(alpha * alpha + sigma)

        # DLARFG formula from LAPACK
        {beta, scale} =
          if alpha <= 0.0 do
            {alpha - norm, 1.0}
          else
            {-sigma / (alpha + norm), norm}
          end

        # Compute v(2:n) = tail / scale
        v_tail = Nx.divide(tail, scale)
        v = Nx.concatenate([Nx.tensor([1.0]), v_tail])

        # tau = 2 * v(1)^2 / (v'v) = 2 / (1 + v_tail'v_tail)
        vn = Nx.sum(Nx.pow(Nx.abs(v_tail), 2)) |> Nx.to_number()
        tau = 2.0 / (1.0 + vn)

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
    n = Nx.size(v)
    m = div(Nx.size(a), n)
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
