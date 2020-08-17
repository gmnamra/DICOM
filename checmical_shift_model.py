import numpy as np

# Find all local minima of discretely evaluated function f(t) with period T
def findMinima(f): return np.where((f < np.roll(f, 1))*(f < np.roll(f, -1)))[0]


# In each voxel, find two smallest local residual minima in a period of omega
def findTwoSmallestMinima(J):
    nVxl = J.shape[1]
    A = np.zeros(nVxl, dtype=int)
    B = np.zeros(nVxl, dtype=int)
    for i in range(nVxl):
        minima = sorted(findMinima(J[:, i]), key=lambda x: J[x, i])[:2]
        if len(minima) == 2:
            A[i], B[i] = minima
        elif len(minima) == 1:
            A[i] = B[i] = minima[0]
        else:
            A[i] = B[i] = 0  # Assign dummy minimum
    return A, B


def getIndexImages(nx, ny, nz):
    left = np.zeros((nz, ny, nx), dtype=bool)
    left[:, :, :-1] = True
    right = np.zeros((nz, ny, nx), dtype=bool)
    right[:, :, 1:] = True
    down = np.zeros((nz, ny, nx), dtype=bool)
    down[:, :-1, :] = True
    up = np.zeros((nz, ny, nx), dtype=bool)
    up[:, 1:, :] = True
    below = np.zeros((nz, ny, nx), dtype=bool)
    below[:-1, :, :] = True
    above = np.zeros((nz, ny, nx), dtype=bool)
    above[1:, :, :] = True
    return (left.flatten(), right.flatten(), up.flatten(), down.flatten(),
            above.flatten(), below.flatten())


# 2D measure of isotropy defined as
# the square area over the square perimeter (area normalized to 1)
def isotropy2D(dx, dy): return np.sqrt(dx*dy)/(2*(dx+dy))


# 3D measure of isotropy defined as
# the cube volume over the cube area (volume normalized to 1)
def isotropy3D(dx, dy, dz): return (dx*dy*dz)**(2/3)/(2*(dx*dy+dx*dz+dy*dz))


def getHigherLevel(level):
    high = {'L': level['L']+1}
    # Isotropy promoting downsampling
    maxIsotropy = 0
    for sx in [1, 2]:
        for sy in [1, 2]:
            for sz in [1, 2]:  # Loop over all 2^3=8 downscaling combinations
                # at least one dimension must change and the size of all
                # dimensions at lower level must permit any downscaling
                if (sx*sy*sz > 1 and level['nx'] >= sx and
                   level['ny'] >= sy and level['nz'] >= sz):
                    if (level['nx'] == 1):
                        iso = isotropy2D(level['dy']*sy, level['dz']*sz)
                    elif (level['ny'] == 1):
                        iso = isotropy2D(level['dx']*sx, level['dz']*sz)
                    elif (level['nz'] == 1):
                        iso = isotropy2D(level['dx']*sx, level['dy']*sy)
                    else:
                        iso = isotropy3D(
                          level['dx']*sx, level['dy']*sy, level['dz']*sz)
                    if iso > maxIsotropy:
                        maxIsotropy = iso
                        high['sx'] = sx
                        high['sy'] = sy
                        high['sz'] = sz
    high['dx'] = level['dx']*high['sx']
    high['dy'] = level['dy']*high['sy']
    high['dz'] = level['dz']*high['sz']

    high['nx'] = int(np.ceil(level['nx']/high['sx']))
    high['ny'] = int(np.ceil(level['ny']/high['sy']))
    high['nz'] = int(np.ceil(level['nz']/high['sz']))
    return high


def getHighLevelResidualImage(J, high, level):
    Jlow = np.zeros((J.shape[0], level['nz']+level['nz'] % high['sz'],
                     level['ny']+level['ny'] % high['sy'],
                     level['nx']+level['nx'] % high['sx']))
    Jlow[:, :level['nz'], :level['ny'], :level['nx']] = J.reshape(
        J.shape[0], level['nz'], level['ny'], level['nx'])

    Jhigh = np.zeros((J.shape[0], high['nz'], high['ny'], high['nx']))

    Jhigh = Jlow[:, ::high['sz'], ::high['sy'], ::high['sx']]
    if high['sx'] > 1:
        Jhigh += Jlow[:, ::high['sz'], ::high['sy'], 1::high['sx']]
    if high['sy'] > 1:
        Jhigh += Jlow[:, ::high['sz'], 1::high['sy'], ::high['sx']]
    if high['sz'] > 1:
        Jhigh += Jlow[:, 1::high['sz'], ::high['sy'], ::high['sx']]
    if high['sx'] > 1 and high['sy'] > 1:
        Jhigh += Jlow[:, ::high['sz'], 1::high['sy'], 1::high['sx']]
    if high['sx'] > 1 and high['sz'] > 1:
        Jhigh += Jlow[:, 1::high['sz'], ::high['sy'], 1::high['sx']]
    if high['sy'] > 1 and high['sz'] > 1:
        Jhigh += Jlow[:, 1::high['sz'], 1::high['sy'], ::high['sx']]
    if high['sx'] > 1 and high['sy'] > 1 and high['sz'] > 1:
        Jhigh += Jlow[:, 1::high['sz'], 1::high['sy'], 1::high['sx']]

    # scale result
    return Jhigh.reshape(Jhigh.shape[0], -1)/(high['sx']*high['sy']*high['sz'])


def getB0fromHighLevel(dB0high, level, high):
    dB0 = np.empty((high['nz']*high['sz'], high['ny']*high['sy'],
                    high['nx']*high['sx']), dtype=int)
    dB0[::high['sz'], ::high['sy'], ::high['sx']] = dB0high
    if high['sx'] > 1:
        dB0[::high['sz'], ::high['sy'], 1::high['sx']] = dB0high
    if high['sy'] > 1:
        dB0[::high['sz'], 1::high['sy'], ::high['sx']] = dB0high
    if high['sz'] > 1:
        dB0[1::high['sz'], ::high['sy'], ::high['sx']] = dB0high
    if high['sx'] > 1 and high['sy'] > 1:
        dB0[::high['sz'], 1::high['sy'], 1::high['sx']] = dB0high
    if high['sx'] > 1 and high['sz'] > 1:
        dB0[1::high['sz'], ::high['sy'], 1::high['sx']] = dB0high
    if high['sy'] > 1 and high['sz'] > 1:
        dB0[1::high['sz'], 1::high['sy'], ::high['sx']] = dB0high
    if high['sx'] > 1 and high['sy'] > 1 and high['sz'] > 1:
        dB0[1::high['sz'], 1::high['sy'], 1::high['sx']] = dB0high
    return dB0[:level['nz'], :level['ny'], :level['nx']].flatten()


# Calculate initial phase phi according to
# Bydder et al. MRI 29 (2011): 216-221.
def getPhi(Y, D):
    phi = np.zeros((Y.shape[1]))
    for i in range(Y.shape[1]):
        y = Y[:, i]
        phi[i] = .5*np.angle(np.dot(np.dot(y.transpose(), D), y))
    return phi


# Calculate phi, remove it from Y and return separate real and imag parts
def getRealDemodulated(Y, D):
    phi = getPhi(Y, D)
    y = Y/np.exp(1j*phi)
    return np.concatenate((np.real(y), np.imag(y))), phi


# Calculate LS error J as function of B0
def getB0Residuals(Y, C, nB0, nVxl, iR2cand, D=None):
    J = np.zeros(shape=(nB0, nVxl))
    # TODO: loop over all R2candidates
    r = 0
    for b in range(nB0):
        if not D:  # complex-valued estimates
            y = Y
        else:  # real-valued estimates
            y, phi = getRealDemodulated(Y, D[r][b])
        J[b, :] = np.linalg.norm(np.dot(C[iR2cand[r]][b], y), axis=0)**2
    return J


# Construct modulation vectors for each B0 value
def modulationVectors(nB0, N):
    B, Bh = [], []
    for b in range(nB0):
        omega = 2.*np.pi*b/nB0
        B.append(np.eye(N)+0j*np.eye(N))
        for n in range(N):
            B[b][n, n] = np.exp(complex(0., n*omega))
        Bh.append(B[b].conj())
    return B, Bh


# Construct matrix RA
def modelMatrix(dPar, mPar, R2):
    RA = np.zeros(shape=(dPar.N, mPar.M))+1j*np.zeros(shape=(dPar.N, mPar.M))
    for n in range(dPar.N):
        t = dPar.t1+n*dPar.dt
        RA[n, 0] = np.exp(complex(-(t-dPar.t1)*R2, 0))  # Water resonance
        for p in range(1, mPar.P):  # Loop over fat resonances
            # Chemical shift between water and peak m (in ppm)
            omega = 2.*np.pi*gyro*dPar.B0*(mPar.CS[p]-mPar.CS[0])
            RA[n, 1] += mPar.alpha[1][p]*np.exp(complex(-(t-dPar.t1)*R2, t*omega))
    return RA


# Get matrix Dtmp defined so that D = Bconj*Dtmp*Bh
# Following Bydder et al. MRI 29 (2011): 216-221.
def getDtmp(A):
    Ah = A.conj().T
    inv = np.linalg.inv(np.real(np.dot(Ah, A)))
    Dtmp = np.dot(A.conj(), np.dot(inv, Ah))
    return Dtmp


# Separate and concatenate real and imag parts of complex matrix M
def realify(M):
    R = np.real(M)
    I = np.imag(M)
    return np.concatenate((np.concatenate((R, I)), np.concatenate((-I, R))), 1)


# Get mean square signal magnitude within foreground
def getMeanEnergy(Y):
    energy = np.linalg.norm(Y, axis=0)**2
    thres = threshold_otsu(energy)
    return np.mean(energy[energy >= thres])


# Perform the actual reconstruction
def reconstruct(dPar, aPar, mPar, B0map=None, R2map=None):
    determineB0 = aPar.graphcutLevel < 20 or aPar.nICMiter > 0
    nR2 = aPar.nR2
    determineR2 = nR2 > 1
    if (nR2 < 0):
        nR2 = -nR2  # nR2<(-1) will use input R2map

    nVxl = dPar.nx*dPar.ny*dPar.nz

    Y = dPar.img
    Y.shape = (dPar.N, nVxl)

    # Prepare matrices
    # Off-resonance modulation vectors (one for each off-resonance value)
    B, Bh = modulationVectors(aPar.nB0, dPar.N)
    RA, RAp, C, Qp = [], [], [], []
    D = None
    if aPar.realEstimates:
        D = []  # Matrix for calculating phi (needed for real-valued estimates)
    for r in range(nR2):
        R2 = r*aPar.R2step
        RA.append(modelMatrix(dPar, mPar, R2))
        if aPar.realEstimates:
            D.append([])
            Dtmp = getDtmp(RA[r])
            for b in range(aPar.nB0):
                D[r].append(np.dot(B[b].conj(), np.dot(Dtmp, Bh[b])))
            RA[r] = np.concatenate((np.real(RA[r]), np.imag(RA[r])))
        RAp.append(np.linalg.pinv(RA[r]))

    if aPar.realEstimates:
        for b in range(aPar.nB0):
            B[b] = realify(B[b])
            Bh[b] = realify(Bh[b])
    for r in range(nR2):
        C.append([])
        Qp.append([])
        # Null space projection matrix
        proj = np.eye(dPar.N*(1+aPar.realEstimates))-np.dot(RA[r], RAp[r])
        for b in range(aPar.nB0):
            C[r].append(np.dot(np.dot(B[b], proj), Bh[b]))
            Qp[r].append(np.dot(RAp[r], Bh[b]))

    # For B0 index -> off-resonance in ppm
    B0step = 1.0/aPar.nB0/dPar.dt/gyro/dPar.B0
    if determineB0:
        V = []  # Precalculate discontinuity costs
        for b in range(aPar.nB0):
            V.append(min(b**2, (b-aPar.nB0)**2))
        V = np.array(V)

        level = {'L': 0, 'nx': dPar.nx, 'ny': dPar.ny, 'nz': dPar.nz,
                 'sx': 1, 'sy': 1, 'sz': 1,
                 'dx': dPar.dx, 'dy': dPar.dy, 'dz': dPar.dz}
        J = getB0Residuals(Y, C, aPar.nB0, nVxl, aPar.iR2cand, D)
        offresPenalty = aPar.offresPenalty
        if aPar.offresPenalty > 0:
            offresPenalty *= getMeanEnergy(Y)

        dB0 = calculateFieldMap(aPar.nB0, level, aPar.graphcutLevel,
                                aPar.multiScale, aPar.maxICMupdate,
                                aPar.nICMiter, J, V, aPar.mu,
                                offresPenalty, int(dPar.offresCenter/B0step))
    elif B0map is None:
        dB0 = np.zeros(nVxl, dtype=int)
    else:
        dB0 = np.array(B0map/B0step, dtype=int)

    if determineR2:
        J = getR2Residuals(Y, dB0, C, aPar.nB0, nR2, nVxl, D)
        R2 = greedyR2(J, nVxl)

    # Find least squares solution given dB0 and R2
    rho = np.zeros(shape=(mPar.M, nVxl))+1j*np.zeros(shape=(mPar.M, nVxl))
    for r in range(nR2):
        for b in range(aPar.nB0):
            vxls = (dB0 == b)*(R2 == r)
            if not D:  # complex estimates
                y = Y[:, vxls]
            else:  # real-valued estimates
                y, phi = getRealDemodulated(Y[:, vxls], D[r][b])
            rho[:, vxls] = np.dot(Qp[r][b], y)
            if D:
                #  Assert phi is the phase angle of water
                phi[rho[0, vxls] < 0] += np.pi
                rho[:, vxls] *= np.exp(1j*phi)

    if B0map is None:
        B0map = np.zeros(nVxl, dtype=IMGTYPE)
    if R2map is None:
        R2map = np.empty(nVxl, dtype=IMGTYPE)

    if determineR2:
        R2map[:] = R2*aPar.R2step

    if determineB0:
        B0map[:] = dB0*B0step

    return rho, B0map, R2map


