defmodule NxLinAlg.EigHessenbergTest do
  use ExUnit.Case, async: true

  alias Nx.LinAlg.EigHessenberg

  def assert_all_close(a, b, tol \\ 1.0e-10) do
    diff = Nx.abs(Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64)))
    max_err = Nx.to_number(Nx.reduce_max(diff))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  defp zero_storage(h) do
    {n, _} = Nx.shape(h)
    # Zero out everything below the first subdiagonal (Householder storage)
    mask =
      Nx.tensor(
        for i <- 0..(n - 1) do
          for j <- 0..(n - 1), do: if(i > j + 1, do: 0.0, else: 1.0)
        end,
        type: :f64
      )

    Nx.multiply(h, mask)
  end

  describe "dgehd2/3" do
    test "3x3 matrix" do
      a =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          type: :f64
        )

      {h_full, q, _tau} = EigHessenberg.dgehd2(a)
      h = zero_storage(h_full)

      # Q * H * Q' ≈ A (H excludes Householder storage)
      assert_all_close(a, Nx.dot(Nx.dot(q, h), Nx.transpose(q)), 1.0e-6)

      # Q is orthogonal
      assert_all_close(Nx.dot(Nx.transpose(q), q), Nx.eye(3, type: :f64), 1.0e-6)
    end

    test "5x5 matrix" do
      a =
        Nx.tensor(
          [
            [2.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 2.0]
          ],
          type: :f64
        )

      {h_full, q, _tau} = EigHessenberg.dgehd2(a)
      h = zero_storage(h_full)

      assert_all_close(a, Nx.dot(Nx.dot(q, h), Nx.transpose(q)), 1.0e-6)
      assert_all_close(Nx.dot(Nx.transpose(q), q), Nx.eye(5, type: :f64), 1.0e-6)
    end

    test "10x10 random" do
      data = for _ <- 1..100, do: (:rand.uniform() * 2 - 1) / 1.0
      a = Nx.tensor(data, type: :f64) |> Nx.reshape({10, 10})
      {h_full, q, _tau} = EigHessenberg.dgehd2(a)
      h = zero_storage(h_full)

      assert_all_close(a, Nx.dot(Nx.dot(q, h), Nx.transpose(q)), 1.0e-6)
      assert_all_close(Nx.dot(Nx.transpose(q), q), Nx.eye(10, type: :f64), 1.0e-6)
    end
  end
end
