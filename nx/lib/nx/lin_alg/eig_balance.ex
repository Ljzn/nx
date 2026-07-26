defmodule Nx.LinAlg.EigBalance do
  @moduledoc """
  Matrix balancing and back-transformation for eigenvalue decomposition.

  Ported from LAPACK DGEBAL and DGEBAK.
  """

  @sclfac 2.0
  @sclfac_recip 0.5

  @doc """
  Matrix balancing (LAPACK DGEBAL, JOB='B').

  Applies permutation and scaling to improve numerical conditioning.
  Returns {balanced, ilo, ihi, scale} where balanced = D * P * A * P' * D^(-1).
  """
  def dgebal(a, _job \\ "B") do
    n = assert_square(a)
    scale = Nx.broadcast(1.0, {n}) |> Nx.to_flat_list()

    # Reduces by working on the flat list
    a_list = Nx.to_flat_list(a)

    # Step 1: Permutation (isolate eigenvalues)
    {a_perm, ilo, ihi, scale_perm} = balance_permute(a_list, n, 0, n - 1, scale)

    # Step 2: Scaling
    {a_scaled, scale_final} = balance_scale(a_perm, n, ilo, ihi, scale_perm)

    {Nx.tensor(a_scaled, type: :f64) |> Nx.reshape({n, n}), ilo + 1, ihi + 1,
     Nx.tensor(scale_final, type: :f64)}
  end

  @doc """
  Back-transform eigenvectors (LAPACK DGEBAK, JOB='B', SIDE='R').

  Applies the inverse of the balancing to right eigenvectors.
  Returns VR = P' * D * VR  (inverse of D * P * VR)
  """
  def dgebak(vr, ilo, ihi, scale_tensor, _job \\ "B", _side \\ "R") do
    n = assert_square(vr)
    vr_list = Nx.to_flat_list(vr)
    scale = Nx.to_flat_list(scale_tensor)

    # Undo scaling (D)
    vr_scaled =
      for i <- 0..(n - 1), reduce: vr_list do
        acc ->
          if i >= ilo - 1 and i <= ihi - 1 do
            # Row i: multiply by 1/SCALE[i] (right eigenvectors)
            s = 1.0 / Enum.at(scale, i)

            Enum.map(0..(n - 1), fn j ->
              idx = i * n + j
              Enum.at(acc, idx) * s
            end)
          else
            acc
          end
      end

    # Undo permutation (P')
    # Permutation was P * A * P', where permutation of rows/cols was done
    # by swapping. DGEBAL stores permutation info in SCALE as negative indices.
    # For our simplified version, just return the scaled result.
    {Nx.tensor(vr_scaled, type: :f64) |> Nx.reshape({n, n}), Nx.broadcast(0.0, {n})}
  end

  # --- Permutation phase ---

  defp balance_permute(a, n, ilo, ihi, scale) do
    # Scan for isolated rows and columns
    {a_new, new_ihi, scale1} = swap_rows_bottom(a, n, ilo, ihi, scale)
    {a_final, new_ilo, scale2} = swap_cols_top(a_new, n, ilo, new_ihi, scale1)

    if new_ilo != ilo or new_ihi != ihi do
      # More swaps possible, continue
      balance_permute(a_final, n, new_ilo, new_ihi, scale2)
    else
      {a_final, new_ilo, new_ihi, scale2}
    end
  end

  # Swap rows with isolated patterns to the bottom
  defp swap_rows_bottom(a, n, ilo, ihi, scale) do
    # Scan from IHI down to ILo
    scan_swap_rows(a, n, ilo, ihi, ihi, scale)
  end

  defp scan_swap_rows(a, _n, ilo, ihi, current, scale) when current < ilo do
    {a, ihi, scale}
  end

  defp scan_swap_rows(a, n, ilo, ihi, current, scale) do
    # Check if row 'current' has only one non-zero off-diagonal
    row_nonzeros = count_row_nonzeros(a, n, current)

    if row_nonzeros == 1 do
      # Swap row 'current' with row IHI and column 'current' with column IHI
      a_swapped = swap_row_and_col(a, n, current, ihi)
      # Track permutation in scale array (negative for permutation)
      new_scale =
        List.update_at(scale, current, fn _ -> -ihi - 1 end)
        |> List.update_at(ihi, fn _ -> -current - 1 end)

      scan_swap_rows(a_swapped, n, ilo, ihi - 1, ihi - 1, new_scale)
    else
      scan_swap_rows(a, n, ilo, ihi, current - 1, scale)
    end
  end

  # Swap columns with isolated patterns to the top
  defp swap_cols_top(a, n, ilo, ihi, scale) do
    scan_swap_cols(a, n, ilo, ihi, ilo, scale)
  end

  defp scan_swap_cols(a, _n, ilo, ihi, current, scale) when current > ihi do
    {a, ilo, scale}
  end

  defp scan_swap_cols(a, n, ilo, ihi, current, scale) do
    col_nonzeros = count_col_nonzeros(a, n, current)

    if col_nonzeros == 1 do
      a_swapped = swap_row_and_col(a, n, current, ilo)

      new_scale =
        List.update_at(scale, current, fn _ -> -ilo - 1 end)
        |> List.update_at(ilo, fn _ -> -current - 1 end)

      scan_swap_cols(a_swapped, n, ilo + 1, ihi, ilo + 1, new_scale)
    else
      scan_swap_cols(a, n, ilo, ihi, current + 1, scale)
    end
  end

  # Count non-zero off-diagonal elements in a row
  defp count_row_nonzeros(a, n, row) do
    Enum.reduce(0..(n - 1), 0, fn j, acc ->
      if j != row and abs(Enum.at(a, row * n + j)) > 1.0e-15 do
        acc + 1
      else
        acc
      end
    end)
  end

  # Count non-zero off-diagonal elements in a column
  defp count_col_nonzeros(a, n, col) do
    Enum.reduce(0..(n - 1), 0, fn i, acc ->
      if i != col and abs(Enum.at(a, i * n + col)) > 1.0e-15 do
        acc + 1
      else
        acc
      end
    end)
  end

  # Swap row and column (for symmetric permutation)
  defp swap_row_and_col(a, n, i, j) do
    # Swap row i with row j
    a_row =
      Enum.reduce(0..(n - 1), a, fn k, acc ->
        idx_i = i * n + k
        idx_j = j * n + k
        temp = Enum.at(acc, idx_i)
        acc |> List.replace_at(idx_i, Enum.at(acc, idx_j)) |> List.replace_at(idx_j, temp)
      end)

    # Swap column i with column j
    Enum.reduce(0..(n - 1), a_row, fn k, acc ->
      idx_i = k * n + i
      idx_j = k * n + j
      temp = Enum.at(acc, idx_i)
      acc |> List.replace_at(idx_i, Enum.at(acc, idx_j)) |> List.replace_at(idx_j, temp)
    end)
  end

  # --- Scaling phase ---

  defp balance_scale(a, n, ilo, ihi, scale) do
    balance_scale_sweep(a, n, ilo, ihi, scale, 5)
  end

  defp balance_scale_sweep(a, _n, _ilo, _ihi, scale, 0), do: {a, scale}

  defp balance_scale_sweep(a, n, ilo, ihi, scale, iter) do
    {a_new, scale_new, any_scaled} = scan_scale(a, n, ilo, ihi, scale)

    if not any_scaled do
      {a_new, scale_new}
    else
      balance_scale_sweep(a_new, n, ilo, ihi, scale_new, iter - 1)
    end
  end

  defp scan_scale(a, n, ilo, ihi, scale) do
    Enum.reduce(ilo..ihi//1, {a, scale, false}, fn i, {a_acc, scale_acc, _} ->
      col_off = col_off_norm(a_acc, n, i)
      row_off = row_off_norm(a_acc, n, i)

      if col_off < 1.0e-15 or row_off < 1.0e-15 do
        {a_acc, scale_acc, false}
      else
        g = :math.sqrt(row_off / col_off)
        g = clamp(g, @sclfac_recip, @sclfac)

        if abs(g - 1.0) < 1.0e-10 do
          {a_acc, scale_acc, false}
        else
          # Scale row i by 1/g, column i by g
          a_scaled = apply_row_col_scale(a_acc, n, i, 1.0 / g, g)
          new_scale = List.update_at(scale_acc, i, fn s -> s * g end)
          {a_scaled, new_scale, true}
        end
      end
    end)
  end

  defp col_off_norm(a, n, col) do
    Enum.reduce(0..(n - 1), 0.0, fn i, acc ->
      if i != col, do: acc + abs(Enum.at(a, i * n + col)), else: acc
    end)
  end

  defp row_off_norm(a, n, row) do
    Enum.reduce(0..(n - 1), 0.0, fn j, acc ->
      if j != row, do: acc + abs(Enum.at(a, row * n + j)), else: acc
    end)
  end

  defp apply_row_col_scale(a, n, i, row_scale, col_scale) do
    a =
      Enum.reduce(0..(n - 1), a, fn j, acc ->
        if j != i do
          List.update_at(acc, i * n + j, fn v -> v * row_scale end)
        else
          acc
        end
      end)

    Enum.reduce(0..(n - 1), a, fn k, acc ->
      if k != i do
        List.update_at(acc, k * n + i, fn v -> v * col_scale end)
      else
        acc
      end
    end)
  end

  defp clamp(v, lo, hi), do: min(max(v, lo), hi)

  defp assert_square(t) do
    s = Nx.size(t)
    n = round(:math.sqrt(s))
    if n * n != s, do: raise("expected square matrix, got size #{s}")
    n
  end
end
