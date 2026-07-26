defmodule Nx.LinAlg.EigHessenberg do
  @moduledoc """
  Hessenberg reduction (LAPACK DGEHD2).

  Reduces general matrix to upper Hessenberg form H = Q' * A * Q.
  """

  @doc """
  DGEHD2: Unblocked Hessenberg reduction.

  Returns {H, Q, tau}.
  """
  def dgehd2(a, ilo \\ 0, ihi \\ nil) do
    n = assert_square(a)
    ihi = ihi || n
    a_f64 = Nx.as_type(a, :f64)

    h_list = Nx.to_flat_list(a_f64)
    tau_list = List.duplicate(0.0, n)
    q = Nx.eye(n, type: :f64)

    {h_final, tau_final, q_final} = process_columns(ilo, ihi, n, h_list, tau_list, q)

    h = Nx.tensor(h_final, type: :f64) |> Nx.reshape({n, n})
    tau_vec = Nx.tensor(tau_final, type: :f64)
    {h, q_final, tau_vec}
  end

  defp process_columns(i, ihi, _n, h_acc, tau_acc, q_acc) when i >= ihi - 1 do
    {h_acc, tau_acc, q_acc}
  end

  defp process_columns(i, ihi, n, h0, tau0, q0) do
    # Step 1: Compute Householder reflector for A(i+1:ihi-1, i)
    alpha = Enum.at(h0, (i + 1) * n + i)
    n_vec = ihi - i
    x_vals = for k <- (i + 2)..(ihi - 1)//1, do: Enum.at(h0, k * n + i)

    full_x =
      if x_vals == [] do
        Nx.tensor([alpha], type: :f64)
      else
        tail_t = Nx.tensor(x_vals, type: :f64)
        Nx.concatenate([Nx.tensor([alpha], type: :f64), tail_t])
      end

    {v, tau, beta} = Nx.LinAlg.EigHouseholder.dlarfg(full_x)

    # Store beta and v in the matrix
    h1 = List.replace_at(h0, (i + 1) * n + i, beta)

    h2 =
      if n_vec >= 3 do
        Enum.reduce(0..(n_vec - 3), h1, fn j, acc ->
          val = Nx.to_number(Nx.reshape(v[j + 1], {}))
          List.replace_at(acc, (i + 2 + j) * n + i, val)
        end)
      else
        h1
      end

    t1 = List.replace_at(tau0, i, tau)

    if tau == 0.0 do
      process_columns(i + 1, ihi, n, h2, t1, q0)
    else
      # Right application uses DLARF1F convention (v[0]=1 implicit, v is [1; tail])
      {h3, q1} = apply_right(h2, q0, v, tau, i, ihi, n)

      # Left application uses DLARF1F convention
      h4 = apply_left(h3, v, tau, i, ihi, n)

      process_columns(i + 1, ihi, n, h4, t1, q1)
    end
  end

  defp apply_right(h_list, q, v, tau, i, ihi, n) do
    sub_m = ihi
    sub_n = ihi - i - 1

    if sub_n <= 0 or sub_m <= 0 do
      {h_list, q}
    else
      vals = for r <- 0..(sub_m - 1), c <- (i + 1)..(ihi - 1), do: Enum.at(h_list, r * n + c)
      sub = Nx.tensor(vals, type: :f64) |> Nx.reshape({sub_m, sub_n})
      result = Nx.LinAlg.EigHouseholder.dlarf(v, tau, sub, :right)

      # Store back
      r_list = Nx.to_flat_list(result)

      h_new =
        Enum.reduce(0..(sub_m - 1), h_list, fn r, acc1 ->
          Enum.reduce(0..(sub_n - 1), acc1, fn c, acc2 ->
            List.replace_at(acc2, r * n + (i + 1 + c), Enum.at(r_list, r * sub_n + c))
          end)
        end)

      # Update Q
      q_new = update_q(q, v, tau, i, ihi, n)
      {h_new, q_new}
    end
  end

  defp apply_left(h_list, v, tau, i, ihi, n) do
    sub_m = ihi - i - 1
    sub_n = n - i - 1

    if sub_m <= 0 or sub_n <= 0 do
      h_list
    else
      vals = for r <- (i + 1)..(ihi - 1), c <- (i + 1)..(n - 1), do: Enum.at(h_list, r * n + c)
      sub = Nx.tensor(vals, type: :f64) |> Nx.reshape({sub_m, sub_n})
      result = Nx.LinAlg.EigHouseholder.dlarf(v, tau, sub, :left)

      l_list = Nx.to_flat_list(result)

      Enum.reduce(0..(sub_m - 1), h_list, fn r, acc1 ->
        Enum.reduce(0..(sub_n - 1), acc1, fn c, acc2 ->
          List.replace_at(acc2, (i + 1 + r) * n + (i + 1 + c), Enum.at(l_list, r * sub_n + c))
        end)
      end)
    end
  end

  defp update_q(q, v, tau, i, _ihi, n) do
    q_list = Nx.to_flat_list(q)
    v_list = Nx.to_flat_list(v)
    n_v = Nx.size(v)

    updated =
      for r <- 0..(n - 1), reduce: q_list do
        q_acc ->
          vx =
            Enum.reduce(0..(n_v - 1), 0.0, fn j, acc ->
              col = i + 1 + j

              if col < n and j < length(v_list) do
                acc + Enum.at(v_list, j) * Enum.at(q_acc, r * n + col)
              else
                acc
              end
            end)

          factor = tau * vx

          if abs(factor) < 1.0e-16 do
            q_acc
          else
            Enum.reduce(0..(n_v - 1), q_acc, fn j, acc ->
              col = i + 1 + j

              if col < n and j < length(v_list) do
                List.replace_at(
                  acc,
                  r * n + col,
                  Enum.at(acc, r * n + col) - factor * Enum.at(v_list, j)
                )
              else
                acc
              end
            end)
          end
      end

    Nx.tensor(updated, type: :f64) |> Nx.reshape({n, n})
  end

  defp assert_square(t) do
    s = Nx.size(t)
    n = round(:math.sqrt(s))
    if n * n != s, do: raise("expected square matrix")
    n
  end
end
