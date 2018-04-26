import numpy as np
from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile, KeplerLightCurve
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import optimize
from contextlib import contextmanager
import warnings
import sys
import logging
import os

@contextmanager
def silence():
    '''Suppreses all output'''
    logger = logging.getLogger()
    logger.disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def fit_leastsq(init, datax, datay, function):

    errfunc = lambda p, x, y: function(p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, init, args=(datax, datay), \
                          full_output=1, epsfcn=0.0001)

    if (len(datay) > len(init)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(init))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)
    return pfit_leastsq, perr_leastsq

def run(kic, period, quarter=5, plot=True):
    with silence():
        lcf = KeplerLightCurveFile.from_archive(kic, quarter=quarter)
    lc = lcf.PDCSAP_FLUX
    timejd = lcf.timeobj.jd
    lc.time = timejd
    with silence():
        tpf = KeplerTargetPixelFile.from_archive(kic, quarter=quarter)
    rawflux = tpf.flux #- tpf.flux_bkg
    fin = np.isfinite(lc.flux)
    lc = lc[fin]
    rawflux = rawflux[fin, :, :]
    timejd = timejd[fin]

    nbins = np.min([int(((lc.time[-1] - lc.time[0])/period)//2), 50])
    fold = lc.remove_nans().normalize().fold(period).bin(nbins)
    phase = np.nanmedian(fold.time[fold.flux < np.nanpercentile(fold.flux, 1)]) + 0.25
    fold = lc.remove_nans().normalize().fold(period, phase).bin(nbins)
    model = np.interp(lc.fold(period, phase).time, fold.time, fold.flux)
    fold = lc.fold(period, phase)
    model_x = fold.time

    modela = model[model_x < 0]
    modela_x = model_x[model_x < 0]
    modelb = model[model_x >= 0]
    modelb_x = model_x[model_x >= 0]

    primary_depth, px = np.nanmin(model[model_x < 0]), model_x[model_x < 0][np.nanargmin(model[model_x < 0])]
    secondary_depth, sx = np.nanmin(model[model_x > 0]),model_x[model_x > 0][np.nanargmin(model[model_x > 0])]
    ph = ((lc.time - phase * period) / period) % 1
    ph[ph > 0.5] -= 1
    not_p = (ph < px - 0.1) | (ph > (px + 0.1))
    not_s = (ph < sx - 0.1) | (ph > (sx + 0.1))

    flux = np.copy(rawflux)
    for i in range(flux.shape[1]):
        for j in range(flux.shape[2]):
            l = KeplerLightCurve(timejd, flux[:,i,j])
            fin = np.isfinite(l.flux)
            if not np.any(fin):
                continue
            sm = savgol_filter(l.flux[not_s & not_p & fin], 101, 1)
            f = interp1d(l.time[not_s & not_p & fin][101:-101], sm[101:-101], fill_value='extrapolate')
            trend = f(l.time)
            flux[:,i,j] /= trend

    if plot:
        fig = plt.figure(figsize=(15,8))
        ax = plt.subplot2grid((2,3), (0,2))
        tpf.plot(ax=ax, aperture_mask=tpf.pipeline_mask)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax = plt.subplot2grid((2,3), (0,0), colspan=2)
        fold.plot(ax=ax)
        ax.plot(model_x, model, lw=3, zorder=10)
        ax.set_title('KIC {}   Period: {:0.2} days   Quarter: {}'.format(kic, period, quarter))

    depths = np.zeros((2, flux.shape[1], flux.shape[2]))*np.nan
    depths_err = np.zeros((2, flux.shape[1], flux.shape[2]))*np.nan
    for i in range(flux.shape[1]):
        for j in range(flux.shape[2]):
            l = KeplerLightCurve(timejd, flux[:,i,j])
            if np.nansum(l.flux) == 0:
                depths[:, i, j] *= np.nan
                continue
            fold = l.normalize().fold(period, phase)
            oks  = [(fold.time < 0), fold.time > 0]
            ms = [modela, modelb]
            for idx, ok, m in zip([0,1], oks, ms):
                def func(params):
                    fl = (params[0] * m) - params[0] + params[1]
                    return fl
                pfit, perr = fit_leastsq([1,1], fold.time[ok], fold.flux[ok], func)
                depths[idx, i, j] = pfit[0]
                depths_err[idx, i, j] = perr[0]

    err = (depths_err[1]**2 + depths_err[0]**2)**0.5
    aper = np.all(np.abs(depths - 1) < 0.5, axis=0)
    aper |= tpf.pipeline_mask
    aper = aper.astype(float)
    aper[aper==0] = np.nan
    aper[err > 0.1] = np.nan

    if plot:
        ax = plt.subplot2grid((2,3), (1,0))
        im = ax.imshow(depths[0]*aper, vmin=0.85, vmax=1.15, cmap=plt.get_cmap('coolwarm'))
        ax.set_title('Primary')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
        ax.grid(False)

        ax = plt.subplot2grid((2,3), (1,1))
        im = ax.imshow(depths[1]*aper, vmin=0.85, vmax=1.15, cmap=plt.get_cmap('coolwarm'))
        ax.set_title('Secondary')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        ax = plt.subplot2grid((2,3), (1,2))
        ax.errorbar(100.*(depths[0]*aper).ravel(), 100.*(depths[1]*aper).ravel(),
                     yerr=100.*(depths_err[0]*aper).ravel(), xerr=100.*(depths_err[1]*aper).ravel(), ls='')
        ax.set_ylabel('Secondary Depth (%)')
        ax.set_xlabel('Primary Depth (%)')
        ax.plot([0,200],[0,200], ls='--', color='gray', alpha=0.7)
        ax.set_xlim(0.9*np.nanmin(depths*np.atleast_2d(aper))*100, 1.1*np.nanmax(depths*np.atleast_2d(aper))*100)
        ax.set_ylim(0.9*np.nanmin(depths*np.atleast_2d(aper))*100, 1.1*np.nanmax(depths*np.atleast_2d(aper))*100)
        ax.grid(alpha=0.5)
    return (depths[0]*aper).ravel(), (depths[1]*aper).ravel(),\
            (depths_err[0]*aper).ravel(), (depths_err[1]*aper).ravel()
