defmodule Nx.LinAlg.EigMultiShiftQR do
  @moduledoc """
  Multi-shift QR algorithm for eigenvalue decomposition.

  Ported from LAPACK DLAQR0-5 (aggressive early deflation, multi-shift bulge chasing).
  """

  @doc """
  DLAQR1: Compute scaling of the first column of
    (H - (sr1 + i*si1)*I) * (H - (sr2 + i*si2)*I)
  for 2x2 or 3x3 H.

  Returns v as a list of length N (2 or 3).
  """
  def dlaqr1(n, h, sr1, si1, sr2, si2) do
    cond do
      n == 2 ->
        h00 = h |> Enum.at(0) |> Enum.at(0)
        h01 = h |> Enum.at(0) |> Enum.at(1)
        h10 = h |> Enum.at(1) |> Enum.at(0)
        h11 = h |> Enum.at(1) |> Enum.at(1)

        s = abs(h00 - sr2) + abs(si2) + abs(h10)

        if s == 0.0 do
          [0.0, 0.0]
        else
          h21s = h10 / s
          v1 = h21s * h01 + (h00 - sr1) * ((h00 - sr2) / s) - si1 * (si2 / s)
          v2 = h21s * (h00 + h11 - sr1 - sr2)
          [v1, v2]
        end

      n == 3 ->
        h00 = h |> Enum.at(0) |> Enum.at(0)
        h01 = h |> Enum.at(0) |> Enum.at(1)
        h02 = h |> Enum.at(0) |> Enum.at(2)
        h10 = h |> Enum.at(1) |> Enum.at(0)
        h11 = h |> Enum.at(1) |> Enum.at(1)
        h12 = h |> Enum.at(1) |> Enum.at(2)
        h20 = h |> Enum.at(2) |> Enum.at(0)
        h21 = h |> Enum.at(2) |> Enum.at(1)
        h22 = h |> Enum.at(2) |> Enum.at(2)

        s = abs(h00 - sr2) + abs(si2) + abs(h10) + abs(h20)

        if s == 0.0 do
          [0.0, 0.0, 0.0]
        else
          h21s = h10 / s
          h31s = h20 / s

          v1 = (h00 - sr1) * ((h00 - sr2) / s) - si1 * (si2 / s) + h01 * h21s + h02 * h31s
          v2 = h21s * (h00 + h11 - sr1 - sr2) + h12 * h31s
          v3 = h31s * (h00 + h22 - sr1 - sr2) + h21s * h21

          [v1, v2, v3]
        end

      true ->
        []
    end
  end
end
