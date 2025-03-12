"""

This is a version of pyfftlog by github user prisae, which itself is based 
on the FFTLog Fortran code by Andrew Hamilton.

This version is updated to perform the specific task of converting 
from a 3d power spectrum to a 3d correlation function using typical
cosmological convention.

The updates take heavy inspiration from a similar code in c / c++ by 
github user slosar (Anze Slosar).

This version updates by Danielle Leonard (github c-d-leonard).

Some of the original Fortran code documentation has been removed for brevity,
 for details see http://casa.colorado.edu/~ajsh/FFTLog.
 
 From previous version by prisae:
 
 ####################################################################

`pyfftlog` -- Python version of FFTLog
======================================

This is a Python version of the FFTLog Fortran code by Andrew Hamilton:

Hamilton, A. J. S., 2000, Uncorrelated modes of the non-linear power spectrum:
Monthly Notices of the Royal Astronomical Society, 312, pages 257-284; DOI:
10.1046/j.1365-8711.2000.03071.x; Website of FFTLog:
http://casa.colorado.edu/~ajsh/FFTLog.

The function `scipy.special.loggamma` replaces the file `cdgamma.f` in the
original code, and the functions `rfft` and `irfft` from `scipy.fftpack`
replace the files `drffti.f`, `drfftf.f`, and `drfftb.f` in the original code.

"""
import numpy as np
from scipy.special import gamma, jv
from scipy.fftpack._fftpack import drfft


def fhti(n, l, dlnr, q=0, kr=1, kropt=0):
    """Initialize the working array xsave used by fftl, fht, and fhtq.

    fhti initializes the working array xsave used by fftl, fht, and fhtq.  fhti
    need be called once, whereafter fftl, fht, or fhtq may be called many
    times, as long as n, mu, q, dlnr, and kr remain unchanged. fhti should be
    called each time n, mu, q, dlnr, or kr is changed. The work array xsave
    should not be changed between calls to fftl, fht, or fhtq.

    Parameters
    ----------
    n : int
        Number of points in the array to be transformed; n may be any positive
        integer, but the FFT routines run fastest if n is a product of small
        primes 2, 3, 5.

    dlnr : float
        Separation between natural log of points; dlnr may be positive or
        negative.

    q : float, optional
        Exponent of power law bias; q may be any real number, positive or
        negative.  If in doubt, use q = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing.  Non-zero q may yield better approximations to the
        continuous Hankel transform for some functions.
        Defaults to 0 (unbiased).

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste).

    kropt : int, optional; {0, 1, 2, 3}
        - 0 to use input kr as is (default);
        - 1 to change kr to nearest low-ringing kr, quietly;
        - 2 to change kr to nearest low-ringing kr, verbosely;
        - 3 for option to change kr interactively.

    Returns
    -------
    kr : float, optional
        kr, adjusted depending on kropt.

    xsave : array
        Working array used by fftl, fht, and fhtq. Dimension:
        - for q = 0 (unbiased transform): n+3
        - for q != 0 (biased transform): 1.5*n+4
        If odd, last element is not needed.

    """

    # adjust kr - don't worry about this if kropt = 0.
    if kropt == 0:    # keep kr as is
        pass
    elif kropt == 1:  # change kr to low-ringing kr quietly
        kr = krgood(l, q, dlnr, kr)
    elif kropt == 2:  # change kr to low-ringing kr verbosely
        d = krgood(l, q, dlnr, kr)
        if abs(kr/d - 1) >= 1e-15:
            kr = d
            print(" kr changed to ", kr)
    else:             # option to change kr to low-ringing kr interactively
        d = krgood(l, q, dlnr, kr)
        if abs(kr/d-1.0) >= 1e-15:
            print(" change kr = ", kr)
            print(" to low-ringing kr = ", d)
            go = input("? [CR, y=yes, n=no, x=exit]: ")
            if go.lower() in ['', 'y']:
                kr = d
                print(" kr changed to ", kr)
            elif go.lower() == 'n':
                print(" kr left unchanged at ", kr)
            else:
                print("exit")
                return False

    # return if n is <= 0
    if n <= 0:
        return kr

    # The normal FFT is not initialized here as in the original FFTLog code, as
    # the `scipy.fftpack`-FFT routines `rfft` and `irfft` do that internally.
    # Therefore xsave in `pyfftlog` is 2*n+15 elements shorter, and named
    # xsave to not confuse it with xsave from the FFT.

    if q == 0:  # unbiased case (q = 0)
        ln2kr = np.log(2.0/kr)   
        xp = (l + 1)/2.0  # I don't know what this l is ..
        d = np.pi/(n*dlnr) # pi / largest l in log space ?

        m = np.arange(1, (n+1)/2) # I don't know what this m is?
        y = m*d  # y = m*pi/(n*dlnr) 
        zp = np.log(gamma(xp + 1j*y))
        arg = 2.0*(ln2kr*y + zp.imag)  # Argument of kr^(-2 i y) U_mu(2 i y)

        # Arange xsave: [q, dlnr, kr, cos, sin]
        xsave = np.empty(2*arg.size+3)
        xsave[0] = q
        xsave[1] = dlnr
        xsave[2] = kr
        xsave[3::2] = np.cos(arg)
        xsave[4::2] = np.sin(arg)

        # Altogether 3 + 2*(n/2) elements used for q = 0, which is n+3 for even
        # n, n+2 for odd n.

    else:       # biased case (q != 0)
        ln2 = np.log(2.0)
        ln2kr = np.log(2.0/kr)
        xp = (l + 1 + q)/2.0
        xm = (l + 1 - q)/2.0

        # first element of rest of xsave
        y = 0

        # case where xp or xm is a negative integer
        xpnegi = np.round(xp) == xp and xp <= 0
        xmnegi = np.round(xm) == xm and xm <= 0
        if xpnegi or xmnegi:

            # case where xp and xm are both negative integers
            # U_mu(q) = 2^q Gamma[xp]/Gamma[xm] is finite in this case
            if xpnegi and xmnegi:
                # Amplitude and Argument of U_mu(q)
                amp = np.exp(ln2*q)
                if xp > xm:
                    m = np.arange(1,  np.round(xp - xm)+1)
                    amp *= xm + m - 1
                elif xp < xm:
                    m = np.arange(1,  np.round(xm - xp)+1)
                    amp /= xp + m - 1
                arg = np.round(xp + xm)*np.pi

            else:  # one of xp or xm is a negative integer
                # Transformation is singular if xp is -ve integer, and inverse
                # transformation is singular if xm is -ve integer, but
                # transformation may be well-defined if sum_j a_j = 0, as may
                # well occur in physical cases.  Policy is to drop the
                # potentially infinite constant in the transform.

                if xpnegi:
                    print('fhti: (mu+1+q)/2 =', np.round(xp), 'is -ve integer',
                          ', yields singular transform:\ntransform will omit',
                          'additive constant that is generically infinite,',
                          '\nbut that may be finite or zero if the sum of the',
                          'elements of the input array a_j is zero.')
                else:
                    print('fhti: (mu+1-q)/2 =', np.round(xm), 'is -ve integer',
                          ', yields singular inverse transform:\n inverse',
                          'transform will omit additive constant that is',
                          'generically infinite,\nbut that may be finite or',
                          'zero if the sum of the elements of the input array',
                          'a_j is zero.')
                amp = 0
                arg = 0

        else:  # neither xp nor xm is a negative integer
            zp = np.log(gamma(xp + 1j*y))
            zm = np.log(gamma(xm + 1j*y))

            # Amplitude and Argument of U_mu(q)
            amp = np.exp(ln2*q + zp.real - zm.real)
            # note +Im(zm) to get conjugate value below real axis
            arg = zp.imag + zm.imag

        # first element: cos(arg) = plus / minus 1, sin(arg) = 0
        xsave1 = amp*np.cos(arg)

        # remaining elements of xsave
        d = np.pi/(n*dlnr)
        m = np.arange(1, (n+1)/2)
        y = m*d  # y = m pi/(n dlnr)
        zp = np.log(gamma(xp + 1j*y))
        zm = np.log(gamma(xm + 1j*y))
        # Amplitude and Argument of kr^(-2 i y) U_mu(q + 2 i y)
        amp = np.exp(ln2*q + zp.real - zm.real)
        arg = 2*ln2kr*y + zp.imag + zm.imag

        # Arrange xsave: [q, dlnr, kr, xsave1, cos, sin]
        xsave = np.empty(3*arg.size+4)
        xsave[0] = q
        xsave[1] = dlnr
        xsave[2] = kr
        xsave[3] = xsave1
        xsave[4::3] = amp
        xsave[5::3] = np.cos(arg)
        xsave[6::3] = np.sin(arg)

        # Altogether 3 + 3*(n/2)+1 elements used for q != 0, which is (3*n)/2+4
        # for even n, (3*n)/2+3 for odd n.  For even n, the very last element
        # of xsave [i.e. xsave(3*m+1)=sin(arg) for m=n/2] is not used within
        # FFTLog; if a low-ringing kr is used, this element should be zero.
        # The last element is computed in case somebody wants it.

    return kr, xsave


def fht(a, xsave, tdir=1):
    """
    Parameters
    ----------
    a : array
        Array A(r) to transform: a(j) is A(r_j) at r_j = r_c exp[(j-jc) dlnr],
        where jc = (n+1)/2 = central index of array.

    xsave : array
        Working array set up by fhti.

    tdir : int, optional; {1, -1}
        -  1 for forward transform (default),
        - -1 for backward transform.
        A backward transform (dir = -1) is the same as a forward transform with
        q -> -q, for any kr if n is odd, for low-ringing kr if n is even.

    Returns
    -------
    a : array
        Transformed array Atilde(k): a(j) is Atilde(k_j) at k_j = k_c exp[(j-jc) dlnr].

    """
    fct = a.copy()
    q = xsave[0]
    dlnr = xsave[1]
    kr = xsave[2]

    # a(r) = A(r) (r/rc)^(-dir*q)
    if q != 0:
        #  centre point of array
        jc = np.array((fct.size + 1)/2.0)
        j = np.arange(fct.size)+1
        fct *= np.exp(-tdir*q*(j - jc)*dlnr)

    # transform a(r) -> atilde(k)
    fct = fhtq(fct, xsave, tdir)

    # Atilde(k) = atilde(k) (k rc)^(-dir*q)
    #      = atilde(k) (k/kc)^(-dir*q) (kc rc)^(-dir*q)
    if q != 0:
        lnkr = np.log(kr)
        fct *= np.exp(-tdir*q*((j - jc)*dlnr + lnkr))

    return fct


def fhtq(a, xsave, tdir=1):
    """
    Parameters
    ----------
    a : array
        Periodic array a(r) to transform: a(j) is a(r_j) at r_j = r_c
        exp[(j-jc) dlnr] where jc = (n+1)/2 = central index of array.

    xsave : array
        Working array set up by fhti.

    tdir : int, optional; {1, -1}
        -  1 for forward transform (default),
        - -1 for backward transform.
        A backward transform (dir = -1) is the same as a forward transform with
        q -> -q, for any kr if n is odd, for low-ringing kr if n is even.

    Returns
    -------
    a : array
        Transformed periodic array atilde(k): a(j) is atilde(k_j) at k_j = k_c exp[(j-jc)
        dlnr].

    """
    fct = a.copy()
    q = xsave[0]
    n = fct.size

    # normal FFT
    # fct = rfft(fct)
    # _raw_fft(fct, n, -1, 1, 1, _fftpack.drfft)
    fct = drfft(fct, n, 1, 0)

    m = np.arange(1, n/2, dtype=int)  # index variable
    if q == 0:  # unbiased (q = 0) transform
        # multiply by (kr)^[- i 2 m pi/(n dlnr)] U_mu[i 2 m pi/(n dlnr)]
        ar = fct[2*m-1]
        ai = fct[2*m]
        fct[2*m-1] = ar*xsave[2*m+1] - ai*xsave[2*m+2]
        fct[2*m] = ar*xsave[2*m+2] + ai*xsave[2*m+1]
        # problem(2*m)atical last element, for even n
        if np.mod(n, 2) == 0:
            ar = xsave[-2]
            if (tdir == 1):  # forward transform: multiply by real part
                # Why? See http://casa.colorado.edu/~ajsh/FFTLog/index.html#ure
                fct[-1] *= ar
            elif (tdir == -1):  # backward transform: divide by real part
                # Real part ar can be zero for maximally bad choice of kr.
                # This is unlikely to happen by chance, but if it does, policy
                # is to let it happen.  For low-ringing kr, imaginary part ai
                # is zero by construction, and real part ar is guaranteed
                # nonzero.
                fct[-1] /= ar

    else:  # biased (q != 0) transform
        # multiply by (kr)^[- i 2 m pi/(n dlnr)] U_mu[q + i 2 m pi/(n dlnr)]
        # phase
        ar = fct[2*m-1]
        ai = fct[2*m]
        fct[2*m-1] = ar*xsave[3*m+2] - ai*xsave[3*m+3]
        fct[2*m] = ar*xsave[3*m+3] + ai*xsave[3*m+2]

        if tdir == 1:  # forward transform: multiply by amplitude
            fct[0] *= xsave[3]
            fct[2*m-1] *= xsave[3*m+1]
            fct[2*m] *= xsave[3*m+1]

        elif tdir == -1:  # backward transform: divide by amplitude
            # amplitude of m=0 element
            ar = xsave[3]
            if ar == 0:
                # Amplitude of m=0 element can be zero for some mu, q
                # combinations (singular inverse); policy is to drop
                # potentially infinite constant.
                fct[0] = 0
            else:
                fct[0] /= ar

            # remaining amplitudes should never be zero
            fct[2*m-1] /= xsave[3*m+1]
            fct[2*m] /= xsave[3*m+1]

        # problematical last element, for even n
        if np.mod(n, 2) == 0:
            m = int(n/2)
            ar = xsave[3*m+2]*xsave[3*m+1]
            if tdir == 1:  # forward transform: multiply by real part
                fct[-1] *= ar
            elif (tdir == -1):  # backward transform: divide by real part
                # Real part ar can be zero for maximally bad choice of kr.
                # This is unlikely to happen by chance, but if it does, policy
                # is to let it happen.  For low-ringing kr, imaginary part ai
                # is zero by construction, and real part ar is guaranteed
                # nonzero.
                fct[-1] /= ar

    # normal FFT back
    # fct = irfft(fct)
    # _raw_fft(fct, n, -1, -1, 1, _fftpack.drfft)
    fct = drfft(fct, n, -1, 1)

    # reverse the array and at the same time undo the FFTs' multiplication by n
    # => Just reverse the array, the rest is already done in drfft.
    fct = fct[::-1]

    return fct


def krgood(mu, q, dlnr, kr):
    """Return optimal kr

    Use of this routine is optional.

    Choosing kr so that
        (kr)^(- i pi/dlnr) U_mu(q + i pi/dlnr)
    is real may reduce ringing of the discrete Hankel transform, because it
    makes the transition of this function across the period boundary smoother.

    Parameters
    ----------
    mu : float
        index of J_mu in Hankel transform; mu may be any real number, positive
        or negative.

    q : float
        exponent of power law bias; q may be any real number, positive or
        negative.  If in doubt, use q = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing.  Non-zero q may yield better approximations to the
        continuous Hankel transform for some functions.

    dlnr : float
        separation between natural log of points; dlnr may be positive or
        negative.

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste).

    Returns
    -------
    krgood : float
        low-ringing value of kr nearest to input kr.  ln(krgood) is always
        within dlnr/2 of ln(kr).

    """
    if dlnr == 0:
        return kr

    xp = (mu + 1.0 + q)/2.0
    xm = (mu + 1.0 - q)/2.0
    y = 1j*np.pi/(2.0*dlnr)
    zp = np.log(gamma(xp + y))
    zm = np.log(gamma(xm + y))

    # low-ringing condition is that following should be integral
    arg = np.log(2.0/kr)/dlnr + (zp.imag + zm.imag)/np.pi

    # return low-ringing kr
    return kr*np.exp((arg - np.round(arg))*dlnr)

def call_transform(l, m, k, pk, tdir):
    """
    tdir is 1 for xi -> pk, -1 for pk -> xi.
    To do xi -> pk, pass r as k and xi as pk.
    l should be passed as 0 and m as 2 for 
    the transform between P(k) and xi(r) in cosmological applications.
    """
    
    n = len(k)
    dlogk = (np.log10(max(k)) - np.log10(min(k)))/n
    dlnk = dlogk*np.log(10.0)
    
    (kr, xsave) = fhti(n, l+0.5, dlnk, 0, 1, 1)

    a = k**(m-0.5) * pk
    b = fht(a, xsave, tdir)
    
    nc = (n + 1)/2.0
    logkc = (np.log10(min(k)) + np.log10(max(k)))/2.
    logrc = np.log10(kr) - logkc
    r = 10**(logrc + (np.arange(1, n+1) - nc)*dlogk)
    
    xi = (2.*np.pi*r)**(-1.5) * b;
    
    return r, xi

def pk2xi(k, pk):
    """ Take the 3d power spectrum at k, output the 3d correlation 
    function at r """

    (r, xi) = call_transform(0, 2, k, pk, tdir=-1)

    return (r, xi)

def xi2pk(r, xi):
    """ Take the 3d correlation function at r, output the 3d power
    spectrum at k. """
    (k, b) = call_transform(0, 2, r, xi, tdir = 1)
    pk = b* 8. * np.pi * np.pi * np.pi
    
    return k, pk

