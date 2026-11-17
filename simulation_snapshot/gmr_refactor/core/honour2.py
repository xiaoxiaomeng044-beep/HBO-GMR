import numpy as np


def honour2(
    rossmary,
    rossmaryporo,
    rossmaryz2,
    N,
    sgz1,
    sgsim2,
    DupdateK2,
    nx=19,
    ny=28,
    nz=5,
    well_locations=None,
):
    unie = rossmary.reshape(nx, ny, nz, order="F")
    unieporo = rossmaryporo.reshape(nx, ny, nz, order="F")
    uniez = rossmaryz2.reshape(nx, ny, nz, order="F")

    total_cells = nx * ny * nz
    output = np.zeros((total_cells, N))
    outputporo = np.zeros((total_cells, N))
    outputz = np.zeros((total_cells, N))

    if well_locations is None:
        well_locations = [(9, 21), (8, 16), (16, 10), (10, 23), (14, 11), (16, 21)]

    for j in range(nz):
        well_perm = [unie[iw, jw, j] for iw, jw in well_locations]
        well_poro = [unieporo[iw, jw, j] for iw, jw in well_locations]
        well_z = [uniez[iw, jw, j] for iw, jw in well_locations]

        for i in range(N):
            B = DupdateK2[:, i].reshape(nx, ny, nz, order="F")
            D = sgsim2[:, i].reshape(nx, ny, nz, order="F")
            D2 = sgz1[:, i].reshape(nx, ny, nz, order="F")

            for idx, (iw, jw) in enumerate(well_locations):
                B[iw, jw, j] = well_perm[idx]
                D[iw, jw, j] = well_poro[idx]
                D2[iw, jw, j] = well_z[idx]

            output[:, i] = B.reshape(total_cells, order="F")
            outputporo[:, i] = D.reshape(total_cells, order="F")
            outputz[:, i] = D2.reshape(total_cells, order="F")

    return output, outputporo, outputz
