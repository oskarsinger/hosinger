from data.loaders.readers import from_num as fn
from periodic import CosineLoader as CL
from gaussian import FakePeriodicGaussianLoader as FPGL
from gaussian import GaussianLoader as GL
from gaussian import LinearRegressionGaussianLoader as LRGL

def get_LRGL(
    n, 
    ps, 
    ws=None, 
    noises=None, 
    noisys=False, 
    bias=False):

    inner_loaders = [GL(n, p) for p in ps]

    if ws is None:
        ws = [None] * len(ps)

    if noises is None:
        noises = [None] * len(ps)

    if not noisys:
        noisys = [False] * len(ps)

    info = zip(
        inner_loaders,
        ws,
        noises,
        noisys)
    
    return [LRGL(il, w=w, noise=ne, noisy=ny, bias=bias)
            for (il, w, ne, ny) in info]

def get_FPGL(n, ps, hertzes):

    return [FPGL(n, p, h) 
            for (p, h) in zip(ps, hertzes)]

def get_cosine_loaders(
    ps,
    periods,
    amplitudes,
    phases,
    indexes,
    period_noise=False,
    phase_noise=False,
    amplitude_noise=False):

    lens = set([
        len(ps),
        len(periods),
        len(amplitudes),
        len(phases),
        len(indexes)])

    if not len(lens) == 1:
        raise ValueError(
            'Args periods, amplitudes, and phases must all have same length.')

    loader_info = zip(
        ps,
        periods,
        amplitudes,
        phases,
        indexes)

    return [_get_CL(p, n, per, a, ph, i,
                period_noise, phase_noise, amplitude_noise)
            for (p, per, a, ph, i) in loader_info]

def _get_CL(
    p,
    max_rounds,
    period,
    amplitude,
    phase,
    index,
    period_noise,
    phase_noise,
    amplitude_noise):

    return CL(
        p,
        max_rounds=max_rounds,
        period=period,
        amplitude=amplitude,
        phase=phase,
        index=index,
        period_noise=period_noise,
        phase_noise=phase_noise,
        amplitude_noise=amplitude_noise)
