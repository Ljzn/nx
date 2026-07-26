defmodule Nx.LinAlg.EigGenerateQ do
  @moduledoc """
  Generate orthogonal matrix Q from Householder reflectors.

  Ported from LAPACK DORG2R (unblocked QR Q generation) and DORGHR
  (Hessenberg Q generation).
  """

  @doc """
  DORG2R: Generate Q from QR decomposition reflectors.

  M×N matrix A stores K Householder vectors below the diagonal
  (column i has v at A(i, i)..A(M, i), where v(1)=1 is implicit).
  Tau has length K. Returns M×N Q.

  The result Q = H(1)*H(2)*...*H(k) overwrites the first N columns of A.
  """
  def dorg2r(a, tau) do
    {m, n} = Nx.shape(a)
    k = Nx.size(tau)

    a_list = Nx.to_flat_list(a)

    # Initialize columns K+1:N as identity
    a1 = init_identity_cols(a_list, m, n, k)

    # Apply reflectors in reverse (K down to 1)
    result = apply_reverse(a1, tau, m, n, k)

    Nx.tensor(result, type: :f64) |> Nx.reshape({m, n})
  end

  @doc """
  DORGHR: Generate Q from Hessenberg reduction reflectors.

  A is N×N, the output of DGEHD2 (Householder vectors stored below
  the subdiagonal in columns ILO..IHI-1). Returns N×N orthogonal Q.
  """
  def dorghr(a, tau, ilo, ihi) do
    n = assert_square(a)
    a_list = Nx.to_flat_list(a)

    # Shift Householder vectors right by one column
    a_shifted = shift_vectors(a_list, n, ilo, ihi)

    # Set first ILO and last N-IHI columns to identity
    a_identity = set_identity_blocks(a_shifted, n, ilo, ihi)

    # Fortran: submatrix A(ILO+1:IHI, ILO+1:IHI) has NH = IHI-ILO columns
    # In 0-indexed: rows from ilo+1 to ihi (inclusive), count = ihi-ilo
    nh = ihi - ilo

    if nh > 0 do
      row_off = ilo + 1

      sub_vals =
        for r <- row_off..ihi, c <- row_off..ihi, do: Enum.at(a_identity, r * n + c)

      sub = Nx.tensor(sub_vals, type: :f64) |> Nx.reshape({nh, nh})
      tau_sub = tau[ilo..(ihi - 1)//1]

      q_sub = dorg2r(sub, tau_sub)
      q_list = Nx.to_flat_list(q_sub)

      # Store back to rows ilo+1..ihi, cols ilo+1..ihi
      Enum.reduce(0..(nh - 1), a_identity, fn r, acc ->
        Enum.reduce(0..(nh - 1), acc, fn c, acc2 ->
          List.replace_at(acc2, (row_off + r) * n + (row_off + c), Enum.at(q_list, r * nh + c))
        end)
      end)
    else
      a_identity
    end
    |> then(&(Nx.tensor(&1, type: :f64) |> Nx.reshape({n, n})))
  end

  # --- DORG2R internals ---

  defp apply_reverse(a, tau, m, n, k) do
    Enum.reduce(k..1//-1, a, fn i, a_acc ->
      tau_i = Nx.to_number(Nx.reshape(tau[i - 1], {}))

      a1 =
        if abs(tau_i) >= 1.0e-16 do
          tail =
            if i < m do
              for r <- i..(m - 1)//1, do: Enum.at(a_acc, r * n + (i - 1))
            else
              []
            end

          v_vals = [1.0 | tail]
          v = Nx.tensor(v_vals, type: :f64)

          a_after_dlarf =
            if i < n do
              sub_rows = m - i + 1
              sub_cols = n - i

              sub_vals =
                for r <- (i - 1)..(m - 1), c <- i..(n - 1), do: Enum.at(a_acc, r * n + c)

              sub = Nx.tensor(sub_vals, type: :f64) |> Nx.reshape({sub_rows, sub_cols})
              result = Nx.LinAlg.EigHouseholder.dlarf(v, tau_i, sub, :left)
              r_list = Nx.to_flat_list(result)

              store_submatrix(a_acc, r_list, n, i - 1, i, sub_rows, sub_cols)
            else
              a_acc
            end

          if i < m do
            Enum.reduce(i..(m - 1)//1, a_after_dlarf, fn r, acc ->
              val = Enum.at(v_vals, r - i + 1) * -tau_i
              List.replace_at(acc, r * n + (i - 1), val)
            end)
          else
            a_after_dlarf
          end
        else
          a_acc
        end

      # A(i, i) = 1 - tau_i (unconditional, even when tau=0)
      a2 = List.replace_at(a1, (i - 1) * n + (i - 1), 1.0 - tau_i)

      # Zero out A(1:i-1, i) (unconditional)
      if i > 1 do
        Enum.reduce(0..(i - 2), a2, fn r, acc ->
          List.replace_at(acc, r * n + (i - 1), 0.0)
        end)
      else
        a2
      end
    end)
  end

  defp init_identity_cols(a, m, n, k) do
    Enum.reduce(k..(n - 1)//1, a, fn j, acc ->
      a1 =
        Enum.reduce(0..(m - 1), acc, fn r, acc2 ->
          List.replace_at(acc2, r * n + j, 0.0)
        end)

      List.replace_at(a1, j * n + j, 1.0)
    end)
  end

  defp store_submatrix(a, r_list, n, row_off, col_off, sub_rows, sub_cols) do
    Enum.reduce(0..(sub_rows - 1), a, fn r, acc ->
      Enum.reduce(0..(sub_cols - 1), acc, fn c, acc2 ->
        List.replace_at(
          acc2,
          (row_off + r) * n + (col_off + c),
          Enum.at(r_list, r * sub_cols + c)
        )
      end)
    end)
  end

  # --- DORGHR internals ---

  defp shift_vectors(a, n, ilo, ihi) do
    # Fortran loop: DO J = IHI, ILO+1, -1
    # In 0-indexed: j (column) goes from ihi-1 down to ilo  (since IHI→ihi-1, ILO+1→ilo)
    # But we store ihi as the LAST column (n-1 for full, or n for n-1)
    # Actually, ihi is passed from Elixir as 0-indexed. The Fortran IHI = ihi_elixir + 1.
    # The loop processes Fortran columns IHI down to ILO+1.
    # In 0-indexed: columns ihi_elixir down to ilo_elixir.
    # However, the Fortran IHI goes UP TO N. If ihi_elixir = n, then Fortran IHI = n+1 → out of bounds.
    # So ihi_elixir must be ≤ n-1 (the last valid index).

    # Fortran: J = IHI to ILO+1 (1-indexed, IHI ≤ N, ILO ≥ 1)
    # 0-indexed column j = J-1 goes from IHI-1 down to ILO (inclusive)
    # j_hi = ihi (Fortran IHI-1), j_lo = ilo (Fortran ILO-1... but ILO+1-1 = ILO = ilo_elixir)
    # So j ranges from ihi down to ilo, where ihi is already 0-indexed

    # Fortran: DO J = IHI, ILO+1, -1  (J is 1-indexed column)
    # In 0-indexed: j = J-1 goes from (IHI-1) down to (ILO+1-1) = from ihi down to ilo
    # But j must stay ≥ 0; j-1 must stay ≥ 0 (source column for copy)
    # 0-indexed j goes from ihi down to ilo+1 (since Fortran J=ILO+1 → j=ilo, and j-1=ilo-1 ≥ 0)
    Enum.reduce(ihi..(ilo + 1)//-1, a, fn j, acc ->
      # DO 10 I = 1, J-1 → zero rows above diagonal
      # 0-indexed: rows 0..(J-1-1) = 0..(j-1)
      acc1 =
        if j > 0 do
          Enum.reduce(0..(j - 1), acc, fn r, acc2 ->
            List.replace_at(acc2, r * n + j, 0.0)
          end)
        else
          acc
        end

      # DO 20 I = J+1, IHI → copy vector tail from (J-1) to J
      # Fortran I=J+1→IHI: 0-indexed i from J to IHI-1 = from j+1 to ihi
      acc2 =
        if j < ihi do
          Enum.reduce((j + 1)..ihi//1, acc1, fn r, acc3 ->
            val = Enum.at(acc3, r * n + (j - 1))
            List.replace_at(acc3, r * n + j, val)
          end)
        else
          acc1
        end

      # DO 30 I = IHI+1, N → zero rows beyond IHI
      if ihi < n - 1 do
        Enum.reduce((ihi + 1)..(n - 1)//1, acc2, fn r, acc3 ->
          List.replace_at(acc3, r * n + j, 0.0)
        end)
      else
        acc2
      end
    end)
  end

  defp set_identity_blocks(a, n, ilo, ihi) do
    # Fortran: DO J = 1, ILO (1-indexed)
    # 0-indexed: j from 0 to ilo-1 (since Fortran ILO = ilo+1)
    a1 =
      Enum.reduce(0..(ilo - 1), a, fn j, acc ->
        a1 =
          Enum.reduce(0..(n - 1), acc, fn r, acc2 ->
            List.replace_at(acc2, r * n + j, 0.0)
          end)

        List.replace_at(a1, j * n + j, 1.0)
      end)

    # Fortran: DO J = IHI+1, N (1-indexed)
    # 0-indexed: j from ihi+1 to n-1
    Enum.reduce((ihi + 1)..(n - 1)//1, a1, fn j, acc ->
      a2 =
        Enum.reduce(0..(n - 1), acc, fn r, acc3 ->
          List.replace_at(acc3, r * n + j, 0.0)
        end)

      List.replace_at(a2, j * n + j, 1.0)
    end)
  end

  defp assert_square(t) do
    s = Nx.size(t)
    n = round(:math.sqrt(s))
    if n * n != s, do: raise("expected square matrix")
    n
  end
end
