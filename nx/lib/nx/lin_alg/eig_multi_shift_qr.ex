defmodule Nx.LinAlg.EigMultiShiftQR do
  @moduledoc """
  Multi-shift QR algorithm for eigenvalue decomposition.

  Ported from LAPACK DLAQR0-5.
  """

  # ── Fortran-indexed array helpers ──
  # Fortran uses 1-based indexing: A(i,j) = flat_list[(i-1)*n + (j-1)]

  defp f_get(h, n, i, j), do: Enum.at(h, (i - 1) * n + (j - 1))
  defp f_set(h, n, i, j, v), do: List.replace_at(h, (i - 1) * n + (j - 1), v)
  defp f_inc(h, n, i, j, dv), do: List.update_at(h, (i - 1) * n + (j - 1), &(&1 + dv))

  @doc """
  DLAQR1: Compute scaling of first column of
    (H - (sr1 + i*si1)*I) * (H - (sr2 + i*si2)*I)
  for 2x2 or 3x3 H. Returns v as list of length N.
  """
  def dlaqr1(n, h, sr1, si1, sr2, si2) do
    cond do
      n == 2 ->
        h00 = h |> Enum.at(0) |> Enum.at(0)
        h01 = h |> Enum.at(0) |> Enum.at(1)
        h10 = h |> Enum.at(1) |> Enum.at(0)

        s = abs(h00 - sr2) + abs(si2) + abs(h10)

        if s == 0.0 do
          [0.0, 0.0]
        else
          h21s = h10 / s
          v1 = h21s * h01 + (h00 - sr1) * ((h00 - sr2) / s) - si1 * (si2 / s)
          v2 = h21s * (h00 + (h |> Enum.at(1) |> Enum.at(1)) - sr1 - sr2)
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

  # ──────────────────────────────────────
  #  DLAQR5: multi-shift bulge sweep
  # ──────────────────────────────────────

  @doc """
  DLAQR5: single multi-shift QR sweep (compact non-ACCUM version).

  Ported from LAPACK DLAQR5, simplified to non-accumulating mode.
  Applies NSHFTS shifts to H(KTOP:KBOT,:).
  """
  def dlaqr5(h, z, n, ktop, kbot, nshfts, sr, si, opts \\ []) do
    wantt = opts[:wantt] || false
    wantz = opts[:wantz] || false
    iloz = opts[:iloz] || 1
    ihiz = opts[:ihiz] || n

    if nshfts < 2 or ktop >= kbot do
      {h, z}
    else
      safmin = Nx.LinAlg.EigUtil.dlamch("S")
      ulp = Nx.LinAlg.EigUtil.dlamch("P")
      smlnum = safmin * (n / ulp)

      # Shuffle shifts into pairs
      {sr, si} = shuffle_shifts(sr, si, nshfts)
      ns = nshfts - rem(nshfts, 2)
      nbmpbs = div(ns, 2)

      # V(3, nbmpbs) — reflector storage
      v = List.duplicate([0.0, 0.0, 0.0], nbmpbs)

      # Main INCOL sweep
      incol0 = ktop - 2 * nbmpbs + 1

      {h, z} =
        dlaqr5_sweep(
          h,
          z,
          n,
          ktop,
          kbot,
          sr,
          si,
          nbmpbs,
          wantt,
          wantz,
          iloz,
          ihiz,
          safmin,
          smlnum,
          ulp,
          v,
          incol0
        )

      {h, z}
    end
  end

  defp shuffle_shifts(sr, si, nsft) do
    sr_list = if is_list(sr), do: sr, else: Nx.to_flat_list(sr)
    si_list = if is_list(si), do: si, else: Nx.to_flat_list(si)

    Enum.reduce(1..(nsft - 2)//2, {sr_list, si_list}, fn i, {sr_a, si_a} ->
      if Enum.at(si_a, i - 1) != -Enum.at(si_a, i) do
        sr2 =
          sr_a
          |> List.replace_at(i - 1, Enum.at(sr_a, i))
          |> List.replace_at(i, Enum.at(sr_a, i + 1 - 1))
          |> List.replace_at(i + 1 - 1, Enum.at(sr_a, i - 1))

        si2 =
          si_a
          |> List.replace_at(i - 1, Enum.at(si_a, i))
          |> List.replace_at(i, Enum.at(si_a, i + 1 - 1))
          |> List.replace_at(i + 1 - 1, Enum.at(si_a, i - 1))

        {sr2, si2}
      else
        {sr_a, si_a}
      end
    end)
  end

  defp dlaqr5_sweep(
         h,
         z,
         n,
         ktop,
         kbot,
         sr,
         si,
         nbmpbs,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         smlnum,
         ulp,
         v,
         incol
       )
       when incol > kbot - 2 do
    {h, z}
  end

  defp dlaqr5_sweep(
         h,
         z,
         n,
         ktop,
         kbot,
         sr,
         si,
         nbmpbs,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         smlnum,
         ulp,
         v,
         incol
       ) do
    jtop = if wantt, do: 1, else: ktop

    krcol_max = min(incol + 2 * nbmpbs - 1, kbot - 2)

    {h, z, v} =
      dlaqr5_krcol_loop(
        h,
        z,
        n,
        ktop,
        kbot,
        sr,
        si,
        nbmpbs,
        wantt,
        wantz,
        iloz,
        ihiz,
        smlnum,
        ulp,
        v,
        incol,
        jtop,
        incol,
        krcol_max
      )

    dlaqr5_sweep(
      h,
      z,
      n,
      ktop,
      kbot,
      sr,
      si,
      nbmpbs,
      wantt,
      wantz,
      iloz,
      ihiz,
      safmin,
      smlnum,
      ulp,
      v,
      incol + 2 * nbmpbs
    )
  end

  # ── KRCOL loop ──

  defp dlaqr5_krcol_loop(
         h,
         z,
         _n,
         _ktop,
         _kbot,
         _sr,
         _si,
         _nbmpbs,
         _wantt,
         _wantz,
         _iloz,
         _ihiz,
         _smlnum,
         _ulp,
         v,
         _incol,
         _jtop,
         krcol,
         krcol_max
       )
       when krcol > krcol_max do
    {h, z, v}
  end

  defp dlaqr5_krcol_loop(
         h,
         z,
         n,
         ktop,
         kbot,
         sr,
         si,
         nbmpbs,
         wantt,
         wantz,
         iloz,
         ihiz,
         smlnum,
         ulp,
         v,
         incol,
         jtop,
         krcol,
         krcol_max
       ) do
    mtop = max(1, div(ktop - krcol, 2) + 1)
    mbot = min(nbmpbs, div(kbot - krcol - 1, 2))
    m22 = mbot + 1
    bmp22 = mbot < nbmpbs and krcol + 2 * (m22 - 1) == kbot - 2

    # 2x2 special case
    {h, z, v} =
      if bmp22 do
        k2 = krcol + 2 * (m22 - 1)

        if k2 == ktop - 1 do
          # Create new 2x2 reflector from shifts
          r0 = k2 + 1

          h2 = [
            [f_get(h, n, r0, r0), f_get(h, n, r0, r0 + 1)],
            [f_get(h, n, r0 + 1, r0), f_get(h, n, r0 + 1, r0 + 1)]
          ]

          v_new =
            dlaqr1(
              2,
              h2,
              Enum.at(sr, 2 * m22 - 2),
              Enum.at(si, 2 * m22 - 2),
              Enum.at(sr, 2 * m22 - 1),
              Enum.at(si, 2 * m22 - 1)
            )

          {href, _tau, _beta} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(v_new, type: :f64))
          vf = Nx.to_flat_list(href)
          t1 = Enum.at(vf, 0)
          t2 = t1 * Enum.at(vf, 1)

          # Right update
          {h, _} =
            Enum.reduce(jtop..min(kbot, k2 + 3), {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(acc, n, j, k2 + 2)

              {acc
               |> f_set(n, j, k2 + 1, f_get(acc, n, j, k2 + 1) - refsum * t1)
               |> f_set(n, j, k2 + 2, f_get(acc, n, j, k2 + 2) - refsum * t2), nil}
            end)

          # Left update
          jbl = if wantt, do: n, else: kbot

          {h, _} =
            Enum.reduce((k2 + 1)..jbl, {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, k2 + 1, j) + Enum.at(vf, 1) * f_get(acc, n, k2 + 2, j)

              {acc
               |> f_set(n, k2 + 1, j, f_get(acc, n, k2 + 1, j) - refsum * t1)
               |> f_set(n, k2 + 2, j, f_get(acc, n, k2 + 2, j) - refsum * t2), nil}
            end)

          # Deflation
          h = deflation_2x2(h, n, k2, ktop, kbot, smlnum, ulp)
          # Accumulate Z
          {h, z} =
            if wantz and not false do
              Enum.reduce(iloz..ihiz, {h, z}, fn j, {h_acc, z_acc} ->
                refsum = f_get(z_acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(z_acc, n, j, k2 + 2)

                {h_acc,
                 z_acc
                 |> f_set(n, j, k2 + 1, f_get(z_acc, n, j, k2 + 1) - refsum * t1)
                 |> f_set(n, j, k2 + 2, f_get(z_acc, n, j, k2 + 2) - refsum * t2)}
              end)
            else
              {h, z}
            end

          v = List.replace_at(v, m22 - 1, vf)
          {h, z, v}
        else
          # Existing 2x2 reflector
          beta = f_get(h, n, k2 + 1, k2)
          vt = [beta, f_get(h, n, k2 + 2, k2)]
          {href, _tau, _beta2} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(vt, type: :f64))
          vf = Nx.to_flat_list(href)
          t1 = Enum.at(vf, 0)
          t2 = t1 * Enum.at(vf, 1)
          h = h |> f_set(n, k2 + 1, k2, beta) |> f_set(n, k2 + 2, k2, 0.0)

          {h, _} =
            Enum.reduce(jtop..min(kbot, k2 + 3), {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(acc, n, j, k2 + 2)

              {acc
               |> f_set(n, j, k2 + 1, f_get(acc, n, j, k2 + 1) - refsum * t1)
               |> f_set(n, j, k2 + 2, f_get(acc, n, j, k2 + 2) - refsum * t2), nil}
            end)

          jbl2 = if wantt, do: n, else: kbot

          {h, _} =
            Enum.reduce((k2 + 1)..jbl2, {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, k2 + 1, j) + Enum.at(vf, 1) * f_get(acc, n, k2 + 2, j)

              {acc
               |> f_set(n, k2 + 1, j, f_get(acc, n, k2 + 1, j) - refsum * t1)
               |> f_set(n, k2 + 2, j, f_get(acc, n, k2 + 2, j) - refsum * t2), nil}
            end)

          h = deflation_2x2(h, n, k2, ktop, kbot, smlnum, ulp)

          {h, z} =
            if wantz do
              Enum.reduce(iloz..ihiz, {h, z}, fn j, {h_acc, z_acc} ->
                refsum = f_get(z_acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(z_acc, n, j, k2 + 2)

                {h_acc,
                 z_acc
                 |> f_set(n, j, k2 + 1, f_get(z_acc, n, j, k2 + 1) - refsum * t1)
                 |> f_set(n, j, k2 + 2, f_get(z_acc, n, j, k2 + 2) - refsum * t2)}
              end)
            else
              {h, z}
            end

          v = List.replace_at(v, m22 - 1, vf)
          {h, z, v}
        end
      else
        {h, z, v}
      end

    # 3x3 reflections (from MBOT down to MTOP)
    Enum.reduce(mbot..mtop//-1, {h, z, v}, fn m, {h_acc, z_acc, v_acc} ->
      k = krcol + 2 * (m - 1)
      v_m = Enum.at(v_acc, m - 1)

      {h_mid, vf} =
        if k == ktop - 1 do
          h3 =
            for i <- 0..2 do
              for j <- 0..2, do: f_get(h_acc, n, ktop + i, ktop + j)
            end

          v_init =
            dlaqr1(
              3,
              h3,
              Enum.at(sr, 2 * m - 2),
              Enum.at(si, 2 * m - 2),
              Enum.at(sr, 2 * m - 1),
              Enum.at(si, 2 * m - 1)
            )

          {href, _tau, _beta} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(v_init, type: :f64))
          vf_new = Nx.to_flat_list(href)
          {h_acc, vf_new}
        else
          # Delayed transformation
          t1o = Enum.at(v_m, 0)
          t2o = t1o * Enum.at(v_m, 1)
          t3o = t1o * Enum.at(v_m, 2)
          refsum_d = Enum.at(v_m, 2) * f_get(h_acc, n, k + 3, k + 2)

          h_acc =
            h_acc
            |> f_set(n, k + 3, k, -refsum_d * t1o)
            |> f_set(n, k + 3, k + 1, -refsum_d * t2o)
            |> f_set(n, k + 3, k + 2, f_get(h_acc, n, k + 3, k + 2) - refsum_d * t3o)

          beta = f_get(h_acc, n, k + 1, k)
          v2_in = f_get(h_acc, n, k + 2, k)
          v3_in = f_get(h_acc, n, k + 3, k)

          {href_n, _tau_n, _beta_n} =
            Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor([beta, v2_in, v3_in], type: :f64))

          vf_n = Nx.to_flat_list(href_n)

          # Bulge collapse check
          hk3k = f_get(h_acc, n, k + 3, k)
          hk3k1 = f_get(h_acc, n, k + 3, k + 1)
          hk3k2 = f_get(h_acc, n, k + 3, k + 2)

          if hk3k != 0.0 or hk3k1 != 0.0 or hk3k2 == 0.0 do
            {h_acc
             |> f_set(n, k + 1, k, beta)
             |> f_set(n, k + 2, k, 0.0)
             |> f_set(n, k + 3, k, 0.0), vf_n}
          else
            h3_sub = for(i <- 0..2, do: for(j <- 0..2, do: f_get(h_acc, n, k + 1 + i, k + 1 + j)))

            vt_init =
              dlaqr1(
                3,
                h3_sub,
                Enum.at(sr, 2 * m - 2),
                Enum.at(si, 2 * m - 2),
                Enum.at(sr, 2 * m - 1),
                Enum.at(si, 2 * m - 1)
              )

            {vt_ref, _tau_vt, _beta_vt} =
              Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(vt_init, type: :f64))

            vt_f = Nx.to_flat_list(vt_ref)
            t1n = Enum.at(vt_f, 0)
            t2n = t1n * Enum.at(vt_f, 1)
            t3n = t1n * Enum.at(vt_f, 2)
            refsum_n = f_get(h_acc, n, k + 1, k) + Enum.at(vt_f, 1) * f_get(h_acc, n, k + 2, k)
            fill_tst = abs(f_get(h_acc, n, k + 2, k) - refsum_n * t2n) + abs(refsum_n * t3n)

            diag_sum =
              abs(f_get(h_acc, n, k, k)) + abs(f_get(h_acc, n, k + 1, k + 1)) +
                abs(f_get(h_acc, n, k + 2, k + 2))

            if fill_tst > ulp * diag_sum do
              {h_acc
               |> f_set(n, k + 1, k, beta)
               |> f_set(n, k + 2, k, 0.0)
               |> f_set(n, k + 3, k, 0.0), vf_n}
            else
              {h_acc
               |> f_set(n, k + 1, k, f_get(h_acc, n, k + 1, k) - refsum_n * t1n)
               |> f_set(n, k + 2, k, 0.0)
               |> f_set(n, k + 3, k, 0.0), vt_f}
            end
          end
        end

      # Apply reflector from right
      t1 = Enum.at(vf, 0)
      t2 = t1 * Enum.at(vf, 1)
      t3 = t1 * Enum.at(vf, 2)

      {h_mid, _} =
        Enum.reduce(jtop..min(kbot, k + 3), {h_mid, nil}, fn j, {acc, _} ->
          refsum =
            f_get(acc, n, j, k + 1) + Enum.at(vf, 1) * f_get(acc, n, j, k + 2) +
              Enum.at(vf, 2) * f_get(acc, n, j, k + 3)

          {acc
           |> f_set(n, j, k + 1, f_get(acc, n, j, k + 1) - refsum * t1)
           |> f_set(n, j, k + 2, f_get(acc, n, j, k + 2) - refsum * t2)
           |> f_set(n, j, k + 3, f_get(acc, n, j, k + 3) - refsum * t3), nil}
        end)

      # First column of left update
      refsum_c =
        f_get(h_mid, n, k + 1, k + 1) + Enum.at(vf, 1) * f_get(h_mid, n, k + 2, k + 1) +
          Enum.at(vf, 2) * f_get(h_mid, n, k + 3, k + 1)

      h_mid =
        h_mid
        |> f_set(n, k + 1, k + 1, f_get(h_mid, n, k + 1, k + 1) - refsum_c * t1)
        |> f_set(n, k + 2, k + 1, f_get(h_mid, n, k + 2, k + 1) - refsum_c * t2)
        |> f_set(n, k + 3, k + 1, f_get(h_mid, n, k + 3, k + 1) - refsum_c * t3)

      # Convergence test
      h_mid = if k >= ktop, do: deflation_3x3(h_mid, n, k, ktop, kbot, smlnum, ulp), else: h_mid

      # Accumulate Z
      {h_mid, z_acc} =
        if wantz do
          Enum.reduce(iloz..ihiz, {h_mid, z_acc}, fn j, {h2, z2} ->
            refsum_z =
              f_get(z2, n, j, k + 1) + Enum.at(vf, 1) * f_get(z2, n, j, k + 2) +
                Enum.at(vf, 2) * f_get(z2, n, j, k + 3)

            {h2,
             z2
             |> f_set(n, j, k + 1, f_get(z2, n, j, k + 1) - refsum_z * t1)
             |> f_set(n, j, k + 2, f_get(z2, n, j, k + 2) - refsum_z * t2)
             |> f_set(n, j, k + 3, f_get(z2, n, j, k + 3) - refsum_z * t3)}
          end)
        else
          {h_mid, z_acc}
        end

      v_acc = List.replace_at(v_acc, m - 1, vf)
      {h_mid, z_acc, v_acc}
    end)
    |> then(fn {h_out, z_out, v_out} ->
      # Left updates for subsequent columns
      jbl3 = if wantt, do: n, else: kbot

      h_out =
        Enum.reduce(mbot..mtop//-1, h_out, fn m, acc ->
          k = krcol + 2 * (m - 1)
          v_m2 = Enum.at(v_out, m - 1)
          t1b = Enum.at(v_m2, 0)
          t2b = t1b * Enum.at(v_m2, 1)
          t3b = t1b * Enum.at(v_m2, 2)

          Enum.reduce(max(ktop, krcol + 2 * m)..jbl3, acc, fn j, acc2 ->
            refsum_l =
              f_get(acc2, n, k + 1, j) + Enum.at(v_m2, 1) * f_get(acc2, n, k + 2, j) +
                Enum.at(v_m2, 2) * f_get(acc2, n, k + 3, j)

            acc2
            |> f_set(n, k + 1, j, f_get(acc2, n, k + 1, j) - refsum_l * t1b)
            |> f_set(n, k + 2, j, f_get(acc2, n, k + 2, j) - refsum_l * t2b)
            |> f_set(n, k + 3, j, f_get(acc2, n, k + 3, j) - refsum_l * t3b)
          end)
        end)

      dlaqr5_krcol_loop(
        h_out,
        z_out,
        n,
        ktop,
        kbot,
        sr,
        si,
        nbmpbs,
        wantt,
        wantz,
        iloz,
        ihiz,
        smlnum,
        ulp,
        v_out,
        incol,
        jtop,
        krcol + 1,
        krcol_max
      )
    end)
  end

  # ── Deflation helpers ──

  defp deflation_2x2(h, n, k, ktop, kbot, smlnum, ulp) do
    if k >= ktop do
      hk1k = f_get(h, n, k + 1, k)

      if hk1k != 0.0 do
        tst1 = abs(f_get(h, n, k, k)) + abs(f_get(h, n, k + 1, k + 1))

        tst1 =
          if tst1 == 0.0 do
            if(k >= ktop + 1, do: abs(f_get(h, n, k, k - 1)), else: 0.0) +
              if k <= kbot - 2, do: abs(f_get(h, n, k + 2, k + 1)), else: 0.0
          else
            tst1
          end

        if abs(hk1k) <= max(smlnum, ulp * tst1) do
          h12d = max(abs(hk1k), abs(f_get(h, n, k, k + 1)))
          h21d = min(abs(hk1k), abs(f_get(h, n, k, k + 1)))

          h11d =
            max(
              abs(f_get(h, n, k + 1, k + 1)),
              abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1))
            )

          h22d =
            min(
              abs(f_get(h, n, k + 1, k + 1)),
              abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1))
            )

          scld = h11d + h12d
          tst2 = h22d * (h11d / scld)

          if tst2 == 0.0 or h21d * (h12d / scld) <= max(smlnum, ulp * tst2) do
            f_set(h, n, k + 1, k, 0.0)
          else
            h
          end
        else
          h
        end
      else
        h
      end
    else
      h
    end
  end

  defp deflation_3x3(h, n, k, ktop, kbot, smlnum, ulp) do
    hk1k = f_get(h, n, k + 1, k)

    if hk1k != 0.0 do
      tst1 = abs(f_get(h, n, k, k)) + abs(f_get(h, n, k + 1, k + 1))

      tst1 =
        if tst1 == 0.0 do
          if(k >= ktop + 1, do: abs(f_get(h, n, k, k - 1)), else: 0.0) +
            if(k >= ktop + 2, do: abs(f_get(h, n, k, k - 2)), else: 0.0) +
            if(k >= ktop + 3, do: abs(f_get(h, n, k, k - 3)), else: 0.0) +
            if(k <= kbot - 2, do: abs(f_get(h, n, k + 2, k + 1)), else: 0.0) +
            if(k <= kbot - 3, do: abs(f_get(h, n, k + 3, k + 1)), else: 0.0) +
            if k <= kbot - 4, do: abs(f_get(h, n, k + 4, k + 1)), else: 0.0
        else
          tst1
        end

      if abs(hk1k) <= max(smlnum, ulp * tst1) do
        h12d = max(abs(hk1k), abs(f_get(h, n, k, k + 1)))
        h21d = min(abs(hk1k), abs(f_get(h, n, k, k + 1)))

        h11d =
          max(abs(f_get(h, n, k + 1, k + 1)), abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1)))

        h22d =
          min(abs(f_get(h, n, k + 1, k + 1)), abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1)))

        scld = h11d + h12d
        tst2 = h22d * (h11d / scld)

        if tst2 == 0.0 or h21d * (h12d / scld) <= max(smlnum, ulp * tst2) do
          f_set(h, n, k + 1, k, 0.0)
        else
          h
        end
      else
        h
      end
    else
      h
    end
  end

  # ──────────────────────────────────────
  #  DTREXC + DLAEXC: eigenvalue reordering
  # ──────────────────────────────────────

  @doc """
  DTREXC: reorder diagonal block IFST to position ILST in Schur form T.
  Returns {t, q, info, ifst_out, ilst_out}.
  """
  def dtrexc(t, q, n, ifst_in, ilst_in, compq \\ "V") do
    wantq = compq == "V"
    ifst = ifst_in
    ilst = ilst_in

    # Quick return
    cond do
      n <= 1 ->
        {t, q, 0, ifst, ilst}

      ifst == ilst ->
        {t, q, 0, ifst, ilst}

      true ->
        # Determine IFST block size
        ifst = if ifst > 1 and f_get(t, n, ifst, ifst - 1) != 0.0, do: ifst - 1, else: ifst
        nbf = if ifst < n and f_get(t, n, ifst + 1, ifst) != 0.0, do: 2, else: 1

        # Determine ILST block size
        ilst = if ilst > 1 and f_get(t, n, ilst, ilst - 1) != 0.0, do: ilst - 1, else: ilst
        nbl = if ilst < n and f_get(t, n, ilst + 1, ilst) != 0.0, do: 2, else: 1

        if ifst < ilst do
          ilst2 =
            cond do
              nbf == 2 and nbl == 1 -> ilst - 1
              nbf == 1 and nbl == 2 -> ilst + 1
              true -> ilst
            end

          move_block_down(t, q, n, ifst, nbf, ilst2, wantq)
        else
          move_block_up(t, q, n, ifst, nbf, ilst, wantq)
        end
    end
  end

  defp move_block_down(t, q, n, here, nbf, ilst, wantq) do
    if here >= ilst do
      {t, q, 0, here, ilst}
    else
      # Determine size of next block below
      nbnext = 1

      nbnext =
        if here + nbf + 1 <= n and f_get(t, n, here + nbf + 1, here + nbf) != 0.0,
          do: 2,
          else: nbnext

      {t, q, info} = dlaexc(t, q, n, here, nbf, nbnext, wantq)

      if info != 0 do
        {t, q, info, here, ilst}
      else
        here2 = here + nbnext

        nbf2 =
          if nbf == 2 do
            if f_get(t, n, here2 + 1, here2) == 0.0, do: 3, else: nbf
          else
            nbf
          end

        move_block_down(t, q, n, here2, nbf2, ilst, wantq)
      end
    end
  end

  defp move_block_up(t, q, n, here, nbf, ilst, wantq) do
    if here <= ilst do
      {t, q, 0, here, ilst}
    else
      nbnext = 1
      nbnext = if here >= 3 and f_get(t, n, here - 1, here - 2) != 0.0, do: 2, else: nbnext

      {t, q, info} = dlaexc(t, q, n, here - nbnext, nbnext, nbf, wantq)

      if info != 0 do
        {t, q, info, here, ilst}
      else
        here2 = here - nbnext

        nbf2 =
          if nbf == 2 do
            if f_get(t, n, here2 + 1, here2) == 0.0, do: 3, else: nbf
          else
            nbf
          end

        move_block_up(t, q, n, here2, nbf2, ilst, wantq)
      end
    end
  end

  @doc """
  DLAEXC: swap adjacent blocks of sizes N1 and N2 (1 or 2 each).
  """
  def dlaexc(t, q, n, j1, n1, n2, wantq) do
    cond do
      n1 == 1 and n2 == 1 ->
        swap_1x1_1x1(t, q, n, j1, wantq)

      n1 == 1 and n2 == 2 ->
        swap_1x1_2x2(t, q, n, j1, wantq)

      n1 == 2 and n2 == 1 ->
        swap_2x2_1x1(t, q, n, j1, wantq)

      n1 == 2 and n2 == 2 ->
        swap_2x2_2x2(t, q, n, j1, wantq)

      true ->
        {t, q, 0}
    end
  end

  # DLARTG wrapper
  defp dlartg(f, g) do
    Nx.LinAlg.EigBlas.dlartg(f, g)
  end

  # LAPACK-exact 1x1 ↔ 1x1: DLARTG + DROT
  defp swap_1x1_1x1(t, q, n, j1, wantq) do
    j2 = j1 + 1
    j3 = j1 + 2
    t11 = f_get(t, n, j1, j1)
    t22 = f_get(t, n, j2, j2)
    {cs, sn, _r} = dlartg(f_get(t, n, j1, j2), t22 - t11)

    {t, _} =
      if j3 <= n do
        {Enum.reduce(j3..n, t, fn j, a ->
           t1j = f_get(a, n, j1, j)
           t2j = f_get(a, n, j2, j)
           a |> f_set(n, j1, j, cs * t1j + sn * t2j) |> f_set(n, j2, j, -sn * t1j + cs * t2j)
         end), nil}
      else
        {t, nil}
      end

    t =
      Enum.reduce(1..(j1 - 1), t, fn i, a ->
        ti1 = f_get(a, n, i, j1)
        ti2 = f_get(a, n, i, j2)
        a |> f_set(n, i, j1, cs * ti1 + sn * ti2) |> f_set(n, i, j2, -sn * ti1 + cs * ti2)
      end)

    t = t |> f_set(n, j1, j1, t22) |> f_set(n, j2, j2, t11)

    q =
      if wantq do
        Enum.reduce(1..n, q, fn i, a ->
          qi1 = f_get(a, n, i, j1)
          qi2 = f_get(a, n, i, j2)
          a |> f_set(n, i, j1, cs * qi1 + sn * qi2) |> f_set(n, i, j2, -sn * qi1 + cs * qi2)
        end)
      else
        q
      end

    {t, q, 0}
  end

  # DLASY2-based swaps (LAPACK-exact)
  defp swap_1x1_2x2(t, q, n, j1, wantq) do
    j2 = j1 + 1
    j3 = j1 + 2
    nd = 3
    d = for i <- 1..nd, j <- 1..nd, do: f_get(t, n, j1 + i - 1, j1 + j - 1)
    dnorm = Enum.reduce(d, 0.0, &max(abs(&1), &2))
    eps = Nx.LinAlg.EigUtil.dlamch("P")
    smlnum = Nx.LinAlg.EigUtil.dlamch("S") / eps
    thresh = max(10.0 * eps * dnorm, smlnum)
    tl = [Enum.at(d, 0)]
    tr = [Enum.at(d, 4), Enum.at(d, 5), Enum.at(d, 7), Enum.at(d, 8)]
    b12 = [Enum.at(d, 1), Enum.at(d, 2)]
    {scale, xf, _, _} = dlasy2(false, false, -1, 1, 2, tl, tr, b12)
    u = [scale, Enum.at(xf, 0), Enum.at(xf, 2)]
    {href, tau, _} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(u, type: :f64))
    uf = Nx.to_flat_list(href)
    uf = [Enum.at(uf, 0), Enum.at(uf, 1), 1.0]
    t11o = f_get(t, n, j1, j1)
    d2 = dla_hh("L", 3, 3, uf, tau, d)
    d2 = dla_hh("R", 3, 3, uf, tau, d2)

    if max(abs(d4_d(d2, 3, 1)), max(abs(d4_d(d2, 3, 2)), abs(d4_d(d2, 3, 3) - t11o))) > thresh,
      do: {t, q, 1}

    t = dla_store(t, d2, n, j1, 3, 3)
    t = t |> f_set(n, j3, j1, 0.0) |> f_set(n, j3, j2, 0.0) |> f_set(n, j3, j3, t11o)
    q = if wantq, do: dla_q(q, n, j1, 3, uf, tau), else: q
    {t, q, 0}
  end

  defp swap_2x2_1x1(t, q, n, j1, wantq) do
    j2 = j1 + 1
    j3 = j1 + 2
    nd = 3
    d = for i <- 1..nd, j <- 1..nd, do: f_get(t, n, j1 + i - 1, j1 + j - 1)
    dnorm = Enum.reduce(d, 0.0, &max(abs(&1), &2))
    eps = Nx.LinAlg.EigUtil.dlamch("P")
    smlnum = Nx.LinAlg.EigUtil.dlamch("S") / eps
    thresh = max(10.0 * eps * dnorm, smlnum)
    tl = [Enum.at(d, 0), Enum.at(d, 1), Enum.at(d, 3), Enum.at(d, 4)]
    tr = [Enum.at(d, 8)]
    b12 = [Enum.at(d, 2), Enum.at(d, 5)]
    {scale, xf, _, _} = dlasy2(false, false, -1, 2, 1, tl, tr, b12)
    u = [-Enum.at(xf, 0), -Enum.at(xf, 1), scale]
    {href, tau, _} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(u, type: :f64))
    uf = Nx.to_flat_list(href)
    uf = [1.0, Enum.at(uf, 1), Enum.at(uf, 2)]
    t33o = f_get(t, n, j3, j3)
    d2 = dla_hh("L", 3, 3, uf, tau, d)
    d2 = dla_hh("R", 3, 3, uf, tau, d2)

    if max(abs(d4_d(d2, 2, 1)), max(abs(d4_d(d2, 3, 1)), abs(d4_d(d2, 1, 1) - t33o))) > thresh,
      do: {t, q, 1}

    t = dla_store(t, d2, n, j1, 3, 3)
    t = t |> f_set(n, j1, j1, t33o) |> f_set(n, j2, j1, 0.0) |> f_set(n, j3, j1, 0.0)
    q = if wantq, do: dla_q(q, n, j1, 3, uf, tau), else: q
    {t, q, 0}
  end

  defp swap_2x2_2x2(t, q, n, j1, wantq) do
    j2 = j1 + 1
    j3 = j1 + 2
    j4 = j1 + 3
    nd = 4
    d = for i <- 1..nd, j <- 1..nd, do: f_get(t, n, j1 + i - 1, j1 + j - 1)
    dnorm = Enum.reduce(d, 0.0, &max(abs(&1), &2))
    eps = Nx.LinAlg.EigUtil.dlamch("P")
    smlnum = Nx.LinAlg.EigUtil.dlamch("S") / eps
    thresh = max(10.0 * eps * dnorm, smlnum)
    tl = [Enum.at(d, 0), Enum.at(d, 1), Enum.at(d, 4), Enum.at(d, 5)]
    tr = [Enum.at(d, 10), Enum.at(d, 11), Enum.at(d, 14), Enum.at(d, 15)]
    b12 = [Enum.at(d, 2), Enum.at(d, 6), Enum.at(d, 3), Enum.at(d, 7)]
    {scale, xf, _, _} = dlasy2(false, false, -1, 2, 2, tl, tr, b12)

    u1 = [-Enum.at(xf, 0), -Enum.at(xf, 1), scale]
    {u1r, tau1, _} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(u1, type: :f64))
    u1f = Nx.to_flat_list(u1r)
    u1f = [1.0, Enum.at(u1f, 1), Enum.at(u1f, 2)]
    temp = -tau1 * (Enum.at(xf, 2) + u1f[1] * Enum.at(xf, 3))
    u2 = [-temp * u1f[1] - Enum.at(xf, 3), -temp * u1f[2], scale]
    {u2r, tau2, _} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(u2, type: :f64))
    u2f = Nx.to_flat_list(u2r)
    u2f = [1.0, Enum.at(u2f, 1), Enum.at(u2f, 2)]

    d2 = dla_hh("L", 3, 4, u1f, tau1, d)
    d2 = dla_hh("R", 4, 3, u1f, tau1, d2)
    d2 = dla_hh("L", 3, 4, u2f, tau2, d2)
    d2 = dla_hh("R", 4, 3, u2f, tau2, d2)

    if max(
         abs(d4_d(d2, 3, 1)),
         max(abs(d4_d(d2, 3, 2)), max(abs(d4_d(d2, 4, 1)), abs(d4_d(d2, 4, 2))))
       ) > thresh, do: {t, q, 1}

    t = dla_store(t, d2, n, j1, 4, 4)

    t =
      t
      |> f_set(n, j3, j1, 0.0)
      |> f_set(n, j3, j2, 0.0)
      |> f_set(n, j4, j1, 0.0)
      |> f_set(n, j4, j2, 0.0)

    q =
      if wantq do
        q = dla_q(q, n, j1, 3, u1f, tau1)
        dla_q(q, n, j2, 3, u2f, tau2)
      else
        q
      end

    {t, q, 0}
  end

  # DLASY2 helpers
  defp dla_hh("L", m, n, v, tau, c) do
    Enum.reduce(0..(n - 1), c, fn j, a ->
      s =
        Enum.reduce(0..(m - 1), 0.0, fn i, acc -> acc + Enum.at(v, i) * d4_d(a, i + 1, j + 1) end)

      t = tau * s

      Enum.reduce(0..(m - 1), a, fn i, a2 ->
        la_set(a2, i * 4 + j, d4_d(a2, i + 1, j + 1) - t * Enum.at(v, i))
      end)
    end)
  end

  defp dla_hh("R", m, n, v, tau, c) do
    Enum.reduce(0..(m - 1), c, fn i, a ->
      s =
        Enum.reduce(0..(n - 1), 0.0, fn j, acc -> acc + d4_d(a, i + 1, j + 1) * Enum.at(v, j) end)

      t = tau * s

      Enum.reduce(0..(n - 1), a, fn j, a2 ->
        la_set(a2, i * 4 + j, d4_d(a2, i + 1, j + 1) - t * Enum.at(v, j))
      end)
    end)
  end

  defp d4_d(f, i, j), do: Enum.at(f, (i - 1) * 4 + (j - 1))
  defp la_set(l, i, v), do: List.replace_at(l, i, v)

  defp dla_store(t, d, n, r0, nr, nc) do
    Enum.reduce(0..(nr - 1), t, fn i, a ->
      Enum.reduce(0..(nc - 1), a, fn j, a2 ->
        f_set(a2, n, r0 + i, r0 + j, Enum.at(d, i * 4 + j))
      end)
    end)
  end

  defp dla_q(q, n, c0, nc, v, tau) do
    Enum.reduce(1..n, q, fn i, a ->
      s =
        Enum.reduce(1..nc, 0.0, fn k, acc ->
          acc + Enum.at(v, k - 1) * f_get(a, n, i, c0 + k - 1)
        end)

      t = tau * s

      Enum.reduce(1..nc, a, fn k, a2 ->
        f_set(a2, n, i, c0 + k - 1, f_get(a2, n, i, c0 + k - 1) - t * Enum.at(v, k - 1))
      end)
    end)
  end

  # ──────────────────────────────────────
  #  DLAQR0: top-level multi-shift QR
  # ──────────────────────────────────────

  @doc """
  DLAQR0: multi-shift QR top-level scheduler.
  Delegates to DLAHQR for matrices N ≤ 75.
  """
  def dlaqr0(h, opts \\ []) do
    wantt = opts[:wantt] || false
    wantz = opts[:wantz] || false
    n = elem(Nx.shape(h), 0)
    ilo = opts[:ilo] || 0
    ihi = opts[:ihi] || n - 1
    iloz = opts[:iloz] || 0
    ihiz = opts[:ihiz] || n - 1

    {h_out, wr, wi, z_out, info} =
      Nx.LinAlg.EigSchur.dlahqr(h,
        wantt: wantt,
        wantz: wantz,
        ilo: ilo,
        ihi: ihi,
        iloz: iloz,
        ihiz: ihiz
      )

    z = if wantz, do: z_out, else: Nx.eye(n, type: :f64)
    {h_out, wr, wi, z, info}
  end

  # ===================== DLASY2 (LAPACK Sylvester solver) =====================

  def dlasy2(ltranl, ltranr, isgn, n1, n2, tl, tr, rhs) do
    eps = Nx.LinAlg.EigUtil.dlamch("P")
    smlnum = Nx.LinAlg.EigUtil.dlamch("S") / eps

    case n1 + n1 + n2 - 2 do
      1 ->
        tau1 = Enum.at(tl, 0) + isgn * Enum.at(tr, 0)
        {tau1, inf} = if abs(tau1) <= smlnum, do: {smlnum, 1}, else: {tau1, 0}
        g = abs(Enum.at(rhs, 0))
        sc = if smlnum * g > abs(tau1), do: 1.0 / g, else: 1.0
        x1 = Enum.at(rhs, 0) * sc / tau1
        {sc, [x1, 0, 0, 0], abs(x1), inf}

      2 ->
        tli = Enum.at(tl, 0)
        t0 = Enum.at(tr, 0)
        t1 = Enum.at(tr, 1)
        t2 = Enum.at(tr, 2)
        t3 = Enum.at(tr, 3)
        a = tli + isgn * t0
        d = tli + isgn * t3
        {b, c} = if ltranr, do: {isgn * t2, isgn * t1}, else: {isgn * t1, isgn * t2}
        smn = max(eps * max(abs(tli), max(abs(t0), max(abs(t1), max(abs(t2), abs(t3))))), smlnum)
        sy2(a, b, c, d, Enum.at(rhs, 0), Enum.at(rhs, 1), smn)

      3 ->
        t0 = Enum.at(tr, 0)
        l0 = Enum.at(tl, 0)
        l1 = Enum.at(tl, 1)
        l2 = Enum.at(tl, 2)
        l3 = Enum.at(tl, 3)
        a = l0 + isgn * t0
        d = l3 + isgn * t0
        {b, c} = if ltranl, do: {l1, l2}, else: {l2, l1}
        smn = max(eps * max(abs(t0), max(abs(l0), max(abs(l1), max(abs(l2), abs(l3))))), smlnum)
        sy2(a, b, c, d, Enum.at(rhs, 0), Enum.at(rhs, 1), smn)

      4 ->
        l0 = Enum.at(tl, 0)
        l1 = Enum.at(tl, 1)
        l2 = Enum.at(tl, 2)
        l3 = Enum.at(tl, 3)
        t0 = Enum.at(tr, 0)
        t1 = Enum.at(tr, 1)
        t2 = Enum.at(tr, 2)
        t3 = Enum.at(tr, 3)
        b11 = Enum.at(rhs, 0)
        b21 = Enum.at(rhs, 1)
        b12 = Enum.at(rhs, 2)
        b22 = Enum.at(rhs, 3)

        smn =
          max(
            eps *
              max(
                abs(t0),
                max(
                  abs(t1),
                  max(abs(t2), max(abs(t3), max(abs(l0), max(abs(l1), max(abs(l2), abs(l3))))))
                )
              ),
            smlnum
          )

        sy4(l0, l1, l2, l3, t0, t1, t2, t3, b11, b21, b12, b22, smn, isgn, ltranl, ltranr, smlnum)
    end
  end

  defp sy2(a, b, c, d, b1, b2, smn) do
    {x1, x2} =
      if abs(a) >= abs(c) do
        m = c / a
        d2 = d - m * b
        b22 = b2 - m * b1
        d2 = if abs(d2) <= smn, do: smn, else: d2
        x2v = b22 / d2
        {(b1 - b * x2v) / a, x2v}
      else
        m = a / c
        bp = b - m * d
        b12 = b1 - m * b2
        bp = if abs(bp) <= smn, do: smn, else: bp
        x2v = b12 / bp
        {(b2 - d * x2v) / c, x2v}
      end

    sc =
      if 2 * smn * abs(x1) > 1 or 2 * smn * abs(x2) > 1,
        do: 0.5 / max(abs(x1), abs(x2)),
        else: 1.0

    {sc, [x1 * sc, 0, x2 * sc, 0], abs(x1 * sc) + abs(x2 * sc), 0}
  end

  defp sy4(a, b, c, d, e, f, g, h, b11, b21, b12, b22, smn, sgn, ltranl, ltranr, sml) do
    t = for _ <- 1..16, do: 0.0
    t = set(t, 0, a + sgn * e)
    t = set(t, 5, d + sgn * e)
    t = set(t, 10, a + sgn * h)
    t = set(t, 15, d + sgn * h)

    t =
      if ltranl,
        do: t |> set(1, c) |> set(4, b) |> set(11, c) |> set(14, b),
        else: t |> set(1, b) |> set(4, c) |> set(11, b) |> set(14, c)

    t =
      if ltranr,
        do: t |> set(2, sgn * f) |> set(7, sgn * f) |> set(8, sgn * g) |> set(13, sgn * g),
        else: t |> set(2, sgn * g) |> set(7, sgn * g) |> set(8, sgn * f) |> set(13, sgn * f)

    b = [b11, b21, b12, b22]
    n = 4
    {t2, b2} = ge4(t, b, smn)

    sc =
      if Enum.any?([0, 1, 2, 3], fn i ->
           8 * sml * abs(Enum.at(b2, i)) > abs(Enum.at(t2, i * 4 + i))
         end) do
        0.125 / Enum.reduce([0, 1, 2, 3], 0.0, fn i, mx -> max(abs(Enum.at(b2, i)), mx) end)
      else
        1.0
      end

    bs = Enum.map(b2, &(&1 * sc))
    xs = [0.0, 0.0, 0.0, 0.0]
    xs = set(xs, 3, Enum.at(bs, 3) / Enum.at(t2, 3 * 4 + 3))

    xs =
      set(
        xs,
        2,
        (Enum.at(bs, 2) - Enum.at(t2, 2 * 4 + 3) * Enum.at(xs, 3)) / Enum.at(t2, 2 * 4 + 2)
      )

    xs =
      set(
        xs,
        1,
        (Enum.at(bs, 1) - Enum.at(t2, 1 * 4 + 2) * Enum.at(xs, 2) -
           Enum.at(t2, 1 * 4 + 3) * Enum.at(xs, 3)) / Enum.at(t2, 1 * 4 + 1)
      )

    xs =
      set(
        xs,
        0,
        (Enum.at(bs, 0) - Enum.at(t2, 0 * 4 + 1) * Enum.at(xs, 1) -
           Enum.at(t2, 0 * 4 + 2) * Enum.at(xs, 2) - Enum.at(t2, 0 * 4 + 3) * Enum.at(xs, 3)) /
          Enum.at(t2, 0 * 4 + 0)
      )

    xn = max(abs(Enum.at(xs, 0)) + abs(Enum.at(xs, 2)), abs(Enum.at(xs, 1)) + abs(Enum.at(xs, 3)))
    {sc, xs, xn, 0}
  end

  defp ge4(t, b, smn) do
    n = 4

    Enum.reduce(0..(n - 2), {t, b}, fn col, {ta, ba} ->
      {pv, _} =
        Enum.reduce(col..(n - 1), {col, -1.0}, fn r, {br, bv} ->
          if abs(Enum.at(ta, r * n + col)) > bv,
            do: {r, abs(Enum.at(ta, r * n + col))},
            else: {br, bv}
        end)

      {ta, ba} =
        if pv != col do
          ta =
            Enum.reduce(0..(n - 1), ta, fn c, m ->
              set(m, col * n + c, Enum.at(m, pv * n + c))
              |> set(pv * n + c, Enum.at(m, col * n + c))
            end)

          {ta, ba |> set(col, Enum.at(ba, pv)) |> set(pv, Enum.at(ba, col))}
        else
          {ta, ba}
        end

      pv2 = abs(Enum.at(ta, col * n + col))
      pv2 = if pv2 < smn, do: smn, else: pv2
      # Fix: use pv2 as the actual value, not as boolean
      piv_val = Enum.at(ta, col * n + col)
      piv_val = if abs(piv_val) < smn, do: smn, else: piv_val

      Enum.reduce((col + 1)..(n - 1), {ta, ba}, fn r, {tb, bb} ->
        mult = Enum.at(tb, r * n + col) / piv_val
        tb = set(tb, r * n + col, mult)
        bb = set(bb, r, Enum.at(bb, r) - mult * Enum.at(bb, col))

        {Enum.reduce((col + 1)..(n - 1), tb, fn c, tc ->
           set(tc, r * n + c, Enum.at(tc, r * n + c) - mult * Enum.at(tc, col * n + c))
         end), bb}
      end)
    end)
  end

  defp set(l, i, v), do: List.replace_at(l, i, v)
end
