import numpy as np
from scipy.optimize.zeros import brentq

def get_Bpe(*, cD=None, Bg=None, Cp=None, csfz=0):
    if Cp is not None:
        return Cp**-(2/3)
    elif (cD is not None) and (Bg is not None):
        return cD**(-0.412/0.764) * Bg**(1/0.382) / (1+2*csfz/cD)**0.5
    else:
        raise ValueError(
            'Insufficient keyword arguments:'
            ' A call to this function must specify either the Cp keyword'
            ' or both the cD and Bg keywords.'
        )

def get_cD(*, Bpe=None, Bg=None, csfz=0):
    if (Bpe is not None) and (Bg is not None):
        return brentq(
            lambda c: Bpe - (
                c**(-0.412/0.764) 
                * Bg**(1/0.382) 
                / (1+2*csfz/c)**0.5
            ),
            1e-9, 1000
        )
    else:
        raise ValueError(
            'Insufficient keyword arguments:'
            ' A call to this function must specify '
            ' both the Bpe and Bg keywords.'
        )

def get_Bg(*, Bpe=None, cD=None, csfz=0, cth=None, Bth=None, Cp=None):
    if Cp is not None:
        return Cp**(1/3-0.588)
    elif (Bpe is not None) and (cD is not None):
        return Bpe**0.382 * cD**0.206 * (1+2*csfz/cD)**0.191
    elif (Bth is not None) and (cth is not None):
        return Bth**1.528 * cth**-0.176
    else:
        raise ValueError(
            'Insufficient keyword arguments:'
            ' A call to this function must specify the Cp keyword,'
            ' both the Bpe and cD keywords, or both the Bth and cth keywords.'
        )

def get_cth(*, Bg=None, Bth=None):
    if (Bg is not None and Bth is not None):
        return Bth**3 * (Bth/Bg)**(1/0.176)
    else:
        raise ValueError(
            'Insufficient keyword arguments:'
            ' A call to this function must specify '
            ' both the Bg and Bth keywords.'
        )

def get_Bth(*, cth=None, Bg=None, Cp=None):
    if Cp is not None:
        return Cp**-(2/3)
    elif (cth is not None) and (Bg is not None):
        return cth**(0.176/1.528) * Bg**(1/1.528)
    else:
        raise ValueError(
            'Insufficient keyword arguments:'
            ' A call to this function must specify either the Cp keyword'
            ' or both the cth and Bg keywords.'
        )

def get_c88(*, Bth=None):
    if Bth is not None:
        return Bth**4
    else:
        raise ValueError(
            'Insufficient keyword arguments:'
            ' A call to this function must specify the Bth keyword.'
        )