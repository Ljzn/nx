defmodule NxLinAlg.EigHouseholderTest do
  use ExUnit.Case, async: true

  alias Nx.LinAlg.EigHouseholder

  def assert_all_close(a, b, tol \\ 1.0e-10) do
    diff = Nx.abs(Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64)))
    max_err = Nx.to_number(Nx.reduce_max(diff))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  describe "dlarfg/1" do
    test "handles positive alpha" do
      x = Nx.tensor([3.0, 4.0], type: :f64)
      {_v, _tau, _beta} = EigHouseholder.dlarfg(x)

      # Just verify it doesn't crash and produces reasonable values
      {v, tau, beta} = EigHouseholder.dlarfg(x)
      n = 2

      h =
        Nx.subtract(
          Nx.eye(n, type: :f64),
          Nx.multiply(tau, Nx.dot(Nx.reshape(v, {n, 1}), Nx.reshape(v, {1, n})))
        )

      hx = Nx.dot(h, x)

      assert abs(Nx.to_number(hx[1])) < 1.0e-6
    end

    test "handles negative alpha" do
      x = Nx.tensor([-3.0, 4.0], type: :f64)
      {v, tau, _beta} = EigHouseholder.dlarfg(x)

      n = 2

      h =
        Nx.subtract(
          Nx.eye(n, type: :f64),
          Nx.multiply(tau, Nx.dot(Nx.reshape(v, {n, 1}), Nx.reshape(v, {1, n})))
        )

      hx = Nx.dot(h, x)

      assert abs(Nx.to_number(hx[1])) < 1.0e-7
    end

    test "reflector is orthogonal" do
      x = Nx.tensor([2.0, -1.0, 3.0, -4.0, 1.0], type: :f64)
      {v, tau, _beta} = EigHouseholder.dlarfg(x)

      n = 5

      h =
        Nx.subtract(
          Nx.eye(n, type: :f64),
          Nx.multiply(tau, Nx.dot(Nx.reshape(v, {n, 1}), Nx.reshape(v, {1, n})))
        )

      hth = Nx.dot(Nx.transpose(h), h)
      assert_all_close(hth, Nx.eye(n, type: :f64), 1.0e-6)
    end

    test "handles scalar (n=1)" do
      x = Nx.tensor([5.0], type: :f64)
      {_v, tau, beta} = EigHouseholder.dlarfg(x)
      assert tau == 0.0
      assert beta == 5.0
    end

    test "v(1) is always 1" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], type: :f64)
      {v, _tau, _beta} = EigHouseholder.dlarfg(x)
      assert Nx.to_number(v[0]) == 1.0
    end
  end

  describe "dlarf/4" do
    test "left application equals explicit H*A" do
      a =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          type: :f64
        )

      x_row = a[0..0] |> Nx.flatten()
      {v, tau, _} = EigHouseholder.dlarfg(x_row)
      n = Nx.size(v)

      # Explicit H
      h =
        Nx.subtract(
          Nx.eye(n, type: :f64),
          Nx.multiply(tau, Nx.dot(Nx.reshape(v, {n, 1}), Nx.reshape(v, {1, n})))
        )

      expected = Nx.dot(h, a)

      result = Nx.LinAlg.EigHouseholder.dlarf(v, tau, a, :left)
      assert_all_close(result, expected)
    end

    test "right application equals explicit A*H" do
      a =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          type: :f64
        )

      x_col = Nx.transpose(a)[0..0] |> Nx.flatten()
      {v, tau, _} = EigHouseholder.dlarfg(x_col)
      n = Nx.size(v)

      h =
        Nx.subtract(
          Nx.eye(n, type: :f64),
          Nx.multiply(tau, Nx.dot(Nx.reshape(v, {n, 1}), Nx.reshape(v, {1, n})))
        )

      expected = Nx.dot(a, h)

      result = Nx.LinAlg.EigHouseholder.dlarf(v, tau, a, :right)
      assert_all_close(result, expected)
    end
  end
end
