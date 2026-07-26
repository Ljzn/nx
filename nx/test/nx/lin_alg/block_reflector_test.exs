defmodule NxLinAlg.EigBlockReflectorTest do
  use ExUnit.Case, async: true

  alias Nx.LinAlg.EigBlockReflector

  def assert_all_close(a, b, tol \\ 1.0e-10) do
    diff = Nx.abs(Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64)))
    max_err = Nx.to_number(Nx.reduce_max(diff))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  describe "dlarft/2" do
    test "k=1 gives tau" do
      v = Nx.tensor([[1.0], [0.5]], type: :f64)
      tau = Nx.tensor([1.6], type: :f64)
      t = EigBlockReflector.dlarft(v, tau)
      assert_all_close(t, Nx.tensor([[1.6]], type: :f64))
    end

    test "k=2 block reflector matches sequential" do
      # Unit lower triangular V (LAPACK convention: column j has j zeros, then 1, then elements)
      # v1 = [1, v21]^T  (for column 0: no leading zeros)
      # v2 = [0, 1, v32]^T  (for column 1: 1 leading zero)... but n=2 so v2 = [0, 1]
      v_mat =
        Nx.tensor([[1.0, 0.0], [0.5, 1.0]],
          type: :f64
        )

      tau_vec = Nx.tensor([1.6, 1.8], type: :f64)

      t_mat = EigBlockReflector.dlarft(v_mat, tau_vec)

      # Full reflector vectors (columns of V)
      # col 0: [1, 0.5]
      v1 = Nx.flatten(Nx.slice(v_mat, [0, 0], [2, 1]))
      # col 1: [0, 1]
      v2 = Nx.flatten(Nx.slice(v_mat, [0, 1], [2, 1]))

      # Sequential: H(1)*H(2) = H1 followed by H2 = H2*H1 (matrix)
      h1 =
        Nx.subtract(
          Nx.eye(2, type: :f64),
          Nx.multiply(1.6, Nx.dot(Nx.reshape(v1, {2, 1}), Nx.reshape(v1, {1, 2})))
        )

      h2 =
        Nx.subtract(
          Nx.eye(2, type: :f64),
          Nx.multiply(1.8, Nx.dot(Nx.reshape(v2, {2, 1}), Nx.reshape(v2, {1, 2})))
        )

      # H_total = H(1)·H(2) = H1*H2 in matrix notation
      h_seq = Nx.dot(h1, h2)

      # Block: H = I - V*T*V'
      h_block =
        Nx.subtract(
          Nx.eye(2, type: :f64),
          Nx.dot(Nx.dot(v_mat, t_mat), Nx.transpose(v_mat))
        )

      assert_all_close(h_block, h_seq, 1.0e-7)
    end
  end

  describe "dlarfb/4" do
    test "left application" do
      v =
        Nx.tensor([[1.0, 0.0], [0.5, 1.0], [0.3, 0.6]],
          type: :f64
        )

      t =
        Nx.tensor([[1.6, 0.5], [0.0, 1.8]],
          type: :f64
        )

      c =
        Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
          type: :f64
        )

      h_block =
        Nx.subtract(
          Nx.eye(3, type: :f64),
          Nx.dot(Nx.dot(v, t), Nx.transpose(v))
        )

      expected = Nx.dot(h_block, c)
      result = EigBlockReflector.dlarfb(v, t, c, :left)
      assert_all_close(result, expected)
    end

    test "right application" do
      v =
        Nx.tensor([[1.0, 0.0], [0.5, 1.0], [0.3, 0.6]],
          type: :f64
        )

      t =
        Nx.tensor([[1.6, 0.5], [0.0, 1.8]],
          type: :f64
        )

      c =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
          type: :f64
        )

      h_block =
        Nx.subtract(
          Nx.eye(3, type: :f64),
          Nx.dot(Nx.dot(v, t), Nx.transpose(v))
        )

      expected = Nx.dot(c, h_block)
      result = EigBlockReflector.dlarfb(v, t, c, :right)
      assert_all_close(result, expected)
    end
  end
end
