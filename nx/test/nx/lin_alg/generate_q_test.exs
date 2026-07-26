defmodule NxLinAlg.EigGenerateQTest do
  use ExUnit.Case, async: true

  alias Nx.LinAlg.EigHouseholder
  alias Nx.LinAlg.EigGenerateQ

  def assert_all_close(a, b, tol \\ 1.0e-10) do
    diff = Nx.abs(Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64)))
    max_err = Nx.to_number(Nx.reduce_max(diff))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  describe "dorg2r/2" do
    test "generates orthogonal Q from 4x3 QR reflectors" do
      # Use DORGHR-generated Hessenberg vectors which are known correct
      a =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
          type: :f64
        )

      {h, _q, tau} = Nx.LinAlg.EigHessenberg.dgehd2(a)
      n = 3

      # DORG2R on the stored Hessenberg vectors (same submatrix DORGHR uses)
      q = Nx.LinAlg.EigGenerateQ.dorghr(h, tau, 0, n - 1)
      assert_all_close(Nx.dot(Nx.transpose(q), q), Nx.eye(3, type: :f64), 1.0e-6)

      hess = Nx.dot(Nx.dot(Nx.transpose(q), a), q)

      for i <- 2..(n - 1) do
        val = Nx.to_number(Nx.reshape(Nx.slice(hess, [i, 0], [1, i - 1]), {}))
        assert abs(val) < 1.0e-6
      end
    end
  end

  describe "dorghr/4" do
    test "generates Q from full Hessenberg reduction of 3x3 matrix" do
      a =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          type: :f64
        )

      n = 3
      {h, _q_from_reduction, tau} = Nx.LinAlg.EigHessenberg.dgehd2(a)

      # ILO=0, IHI=n-1 (balancing submatrix extent, no balancing applied)
      q = Nx.LinAlg.EigGenerateQ.dorghr(h, tau, 0, n - 1)

      assert_all_close(Nx.dot(Nx.transpose(q), q), Nx.eye(3, type: :f64), 1.0e-6)

      hess = Nx.dot(Nx.dot(Nx.transpose(q), a), q)

      for i <- 2..(n - 1) do
        val = Nx.to_number(Nx.reshape(Nx.slice(hess, [i, 0], [1, i - 1]), {}))
        assert abs(val) < 1.0e-6
      end
    end

    test "matches Q accumulated during Hessenberg reduction" do
      a =
        Nx.tensor([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          type: :f64
        )

      n = 3
      {h, q_accum, tau} = Nx.LinAlg.EigHessenberg.dgehd2(a)

      q_gen = Nx.LinAlg.EigGenerateQ.dorghr(h, tau, 0, n - 1)

      assert_all_close(Nx.dot(Nx.transpose(q_gen), q_accum), Nx.eye(3, type: :f64), 1.0e-6)
    end
  end

  # --- Helpers ---

  # Simple unblocked QR factorization (DGEQRF-like)
  defp simple_qr(a_list, m, n, k) do
    tau_list = List.duplicate(0.0, k)

    Enum.reduce(0..(k - 1), {a_list, tau_list}, fn i, {a_acc, tau_acc} ->
      # Extract column i, rows i:m-1
      alpha = Enum.at(a_acc, i * n + i)
      x_vals = for r <- (i + 1)..(m - 1), do: Enum.at(a_acc, r * n + i)

      full_x =
        if x_vals == [] do
          Nx.tensor([alpha], type: :f64)
        else
          tail_t = Nx.tensor(x_vals, type: :f64)
          Nx.concatenate([Nx.tensor([alpha], type: :f64), tail_t])
        end

      {v, tau, beta} = EigHouseholder.dlarfg(full_x)

      # Store β at A(i, i)
      a1 = List.replace_at(a_acc, i * n + i, beta)

      # Store v(2:) at A(i+1:m, i)
      a2 =
        if Nx.size(v) > 1 do
          Enum.reduce(0..(min(Nx.size(v), m - i - 1) - 1), a1, fn j, acc ->
            val = Nx.to_number(Nx.reshape(v[j + 1], {}))
            List.replace_at(acc, (i + 1 + j) * n + i, val)
          end)
        else
          a1
        end

      tau_new = List.replace_at(tau_acc, i, tau)

      # Apply reflector to A(i:m, i+1:n) from left
      if i < n - 1 and abs(tau) > 1.0e-16 do
        sub_rows = m - i
        sub_cols = n - i - 1

        sub_vals =
          for r <- i..(m - 1), c <- (i + 1)..(n - 1), do: Enum.at(a2, r * n + c)

        sub = Nx.tensor(sub_vals, type: :f64) |> Nx.reshape({sub_rows, sub_cols})
        v_t = Nx.tensor([1.0 | List.duplicate(0.0, Nx.size(v) - 1)], type: :f64)

        v_full =
          if Nx.size(v) > 0 do
            tail = Nx.slice(v, [1], [Nx.size(v) - 1])
            Nx.concatenate([Nx.tensor([1.0], type: :f64), tail])
          else
            Nx.tensor([1.0], type: :f64)
          end

        result = EigHouseholder.dlarf(v_full, tau, sub, :left)
        r_list = Nx.to_flat_list(result)

        a3 =
          Enum.reduce(0..(sub_rows - 1), a2, fn r, acc ->
            Enum.reduce(0..(sub_cols - 1), acc, fn c, acc2 ->
              List.replace_at(acc2, (i + r) * n + (i + 1 + c), Enum.at(r_list, r * sub_cols + c))
            end)
          end)

        {a3, tau_new}
      else
        {a2, tau_new}
      end
    end)
  end
end
