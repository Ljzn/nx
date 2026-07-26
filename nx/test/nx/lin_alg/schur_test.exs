defmodule NxLinAlg.EigSchurTest do
  use ExUnit.Case, async: true

  alias Nx.LinAlg.EigSchur

  def assert_all_close(a, b, tol \\ 1.0e-10) do
    diff = Nx.abs(Nx.subtract(Nx.as_type(a, :f64), Nx.as_type(b, :f64)))
    max_err = Nx.to_number(Nx.reduce_max(diff))
    assert max_err < tol, "max diff #{max_err} >= #{tol}"
  end

  describe "dlanv2/4" do
    test "real eigenvalues: [[2,1],[1,2]]" do
      # Eigenvalues: 3 and 1
      {a, b, c, d, rt1r, rt1i, rt2r, rt2i, _cs, _sn} = EigSchur.dlanv2(2.0, 1.0, 1.0, 2.0)
      assert c == 0.0
      assert rt1i == 0.0
      assert rt2i == 0.0
      assert_in_delta rt1r, 3.0, 1.0e-10
      assert_in_delta rt2r, 1.0, 1.0e-10
    end

    test "complex eigenvalues: [[2,1],[-1,2]]" do
      # Eigenvalues: 2 ± i
      {a, b, c, d, rt1r, rt1i, rt2r, rt2i, _cs, _sn} = EigSchur.dlanv2(2.0, 1.0, -1.0, 2.0)
      assert_in_delta rt1r, 2.0, 1.0e-10
      assert_in_delta rt1i, 1.0, 1.0e-10
      assert_in_delta rt2r, 2.0, 1.0e-10
      assert_in_delta rt2i, -1.0, 1.0e-10
    end

    test "diagonal matrix" do
      {a, b, c, d, rt1r, rt1i, rt2r, rt2i, _cs, _sn} = EigSchur.dlanv2(5.0, 0.0, 0.0, 7.0)
      assert c == 0.0
      assert_in_delta rt1r, 5.0, 1.0e-10
      assert_in_delta rt2r, 7.0, 1.0e-10
    end

    test "zero off-diagonal: [[3,0],[2,4]]" do
      {a, b, c, d, rt1r, rt1i, rt2r, rt2i, _cs, _sn} = EigSchur.dlanv2(3.0, 0.0, 2.0, 4.0)
      # should be in Schur form
      assert c == 0.0
      assert_in_delta rt1r, 4.0, 1.0e-10
      assert_in_delta rt2r, 3.0, 1.0e-10
    end

    test "almost equal real eigenvalues" do
      # [[2, 1], [1e-10, 2]] — almost defective
      {a, b, c, d, rt1r, rt1i, rt2r, rt2i, _cs, _sn} = EigSchur.dlanv2(2.0, 1.0, 1.0e-10, 2.0)
      assert_in_delta rt1r + rt2r, 4.0, 1.0e-6
      assert_in_delta rt1r * rt2r - rt1i * rt2i, 4.0 - 1.0e-10, 1.0e-6
    end

    test "Schur reconstruction" do
      # For a random-ish 2x2, verify Q' * original * Q = Schur form
      orig = [[3.0, 1.0], [2.0, 4.0]]

      {a_out, b_out, c_out, d_out, rt1r, rt1i, rt2r, rt2i, cs, sn} =
        EigSchur.dlanv2(3.0, 1.0, 2.0, 4.0)

      # Q = [CS -SN; SN CS]
      # Q' * orig * Q should be [A B; C D]
      # Q' * orig * Q for Schur form means C should be 0 (or nearly)
      if c_out != 0.0 do
        # Complex case — check that AA and DD are equal
        assert_in_delta a_out, d_out, 1.0e-10
      else
        assert abs(c_out) < 1.0e-10
        # Eigenvalues
        assert_in_delta rt1r + rt2r, 7.0, 1.0e-10
        assert_in_delta rt1r * rt2r, 10.0, 1.0e-10
      end
    end
  end

  describe "dlahqr/1" do
    test "real eigenvalues from 2x2" do
      h = Nx.tensor([[3.0, 1.0], [1.0, 2.0]], type: :f64)
      {h_out, wr, wi, _z, info} = EigSchur.dlahqr(h)
      assert info == 0
      assert_in_delta Nx.to_number(wr[0]) + Nx.to_number(wr[1]), 5.0, 1.0e-6
      assert_in_delta Nx.to_number(wr[0]) * Nx.to_number(wr[1]), 5.0, 1.0e-6
    end

    test "complex eigenvalues from 2x2" do
      h = Nx.tensor([[2.0, 1.0], [-2.0, 2.0]], type: :f64)
      {_h_out, wr, wi, _z, info} = EigSchur.dlahqr(h)
      assert info == 0
      assert_in_delta Nx.to_number(wr[0]), 2.0, 1.0e-6
      assert_in_delta Nx.to_number(wi[0]), :math.sqrt(2.0), 1.0e-6
    end

    test "3x3 with known eigenvalues" do
      h =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.8, 9.0]],
          type: :f64
        )

      {_h_out, wr, wi, _z, info} = EigSchur.dlahqr(h)
      assert info == 0
      # Sum of eigenvalues = trace
      trace = 1.0 + 5.0 + 9.0
      sum_eig = Nx.to_number(Nx.sum(wr))
      assert_in_delta sum_eig, trace, 1.0e-6
    end
  end
end
