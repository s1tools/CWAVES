
#__________________________________________
#To be reviewed by Ifremer for confirmation


def get_cwaves_kernels_modifv2(
        cspc_re,
        cspc_im,
        kx,
        ky,
):
    """
    CWAVE v2 python implementation: 
    var: cspc_re : real part cross spectra (ky, kx)
    var: cspc_im : imaginary part of the cross spectra (ky, kx)
    var: kx : wavenumber vector
    var: ky : wavenumber vector
    return :
    """
    kmax = 2 * np.pi / 15  # ~0.104 rad/m (60m wavelength) # ok
    kmin = 2 * np.pi / 625  # ~0.010 rad/m (625m wavelength) #ok
    ns = 20
    S = np.ones((ns, 1)) * np.nan

    cspc_re = cspc_re.T  # converted to (kx, ky)
    cspc_im = cspc_im.T  # converted to (kx, ky)

    KY, KX = np.meshgrid(kx, ky, indexing='ij')  # Shape: (255, 185)

    # True 2D radial wavenumber magnitude
    k_rad = np.sqrt(KX ** 2 + KY ** 2)  # ok
    mask = (k_rad < kmin) | (k_rad > kmax)  # Shape: (kx.size, ky.size)

    # Compute dkx and dky
    dkx = np.zeros(kx.size)
    if kx.size > 1:
        dkx[0] = kx[1] - kx[0]
        dkx[-1] = kx[-1] - kx[-2]
        dkx[1:-1] = (kx[2:] - kx[0:-2]) / 2
    else:
        dkx[0] = 1.0

    dky = np.zeros(ky.size)
    if ky.size > 1:
        dky[0] = ky[1] - ky[0]
        dky[-1] = ky[-1] - ky[-2]
        dky[1:-1] = (ky[2:] - ky[0:-2]) / 2
    else:
        dky[0] = 1.0

    DKY, DKX = np.meshgrid(dkx, dky, indexing='ij')

    subset_ok = pd.DataFrame()

    for jj in range(ns):
        subset_ok.loc[0, 's' + str(jj)] = np.nan

    expected_shape = (kx.size, ky.size)

    h = np.full((kx.size, ky.size, ns), np.nan)

    if (
            all([_itm.shape == expected_shape for _itm in [cspc_re, cspc_im]])
            and (cspc_re > 0).any()
    ):

        try:
            cspc = np.sqrt(cspc_re ** 2 + cspc_im ** 2)  # Shape: (kx.size, ky.size)
            cspc_masked = ma.masked_where(mask, cspc, copy=True).filled(np.nan)  # ok

            gamma = 2.0
            a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)  # ok
            a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (
                    kmax ** 2 - gamma ** 2 * kmin ** 2
            )  # ok

            tmp = a1 * np.power(KX, 4) + a2 * KX ** 2 + KY ** 2  # ok

            qt = (KX ** 2 + KY ** 2) * tmp * np.log10(kmax / kmin)  # ok
            qt_safe = np.where(qt == 0, np.nan, qt)  # ok
            eta = np.sqrt(np.true_divide(2.0 * (2 * a1 * np.power(KX, 4) + a2 * KX ** 2 + KY ** 2), qt_safe))  # ok

            with np.errstate(divide='ignore', invalid='ignore'):
                alphak = (
                        2.0
                        * (
                                (
                                        np.log10(np.maximum(np.sqrt(tmp), 1e-12), where=(tmp > 0))
                                        - np.log10(kmin)
                                )
                                / np.log10(kmax / kmin)
                        )
                        - 1.0
                )  # ok but to confirm log10 ! 

            alphak = np.clip(alphak, -1.0, 1.0)  # ok
            alphat = np.arctan2(KY, KX)  # ok

            g_term = np.sqrt(np.maximum(1.0 - alphak ** 2, 0.0))  # ok

            g1 = 0.5 * np.sqrt(3.0) * g_term  # ok
            g2 = 0.5 * np.sqrt(15.0) * alphak * g_term  # ok
            g3 = (1.0 / 4.0) * np.sqrt(7.0 / 6.0) * (15.0 * alphak ** 2 - 3.0) * g_term  # ok
            g4 = (1.0 / 4.0) * np.sqrt(9.0 / 10.0) * (35.0 * alphak ** 3 - 15.0 * alphak) * g_term  # ok 

            f1 = np.sqrt(1.0 / np.pi)  # ok
            f2 = np.sqrt(2.0 / np.pi) * np.sin(2.0 * alphat)  # ok
            f3 = np.sqrt(2.0 / np.pi) * np.cos(2.0 * alphat)  # ok
            f4 = np.sqrt(2.0 / np.pi) * np.sin(4.0 * alphat)  # ok
            f5 = np.sqrt(2.0 / np.pi) * np.cos(4.0 * alphat)  # ok

            h = np.zeros((KX.shape[0], KX.shape[1], 20))  # ok 
            g_list = [g1, g2, g3, g4]
            f_list = [f1, f2, f3, f4, f5]

            idx = 0
            for g in g_list:
                for f in f_list:
                    h[:, :, idx] = g * f * eta
                    idx += 1

            normalization_denom = np.nansum(cspc_masked * DKX * DKY) #ok
            if normalization_denom > 0:
                P = cspc_masked / normalization_denom
            else:
                P = np.zeros_like(cspc_masked)

            for jj in range(ns):
                petit_h = h[:, :, jj]
                val = np.nansum(petit_h * P * DKX * DKY)  # ok
                S[jj] = val
                subset_ok.loc[0, 's' + str(jj)] = val

        except Exception as e:
            print(f'[Warning] CWAVE extraction failed: {e}')

    return subset_ok, S, h
