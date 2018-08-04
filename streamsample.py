# Third-party

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
import scipy.optimize as so
import emcee

import gala.coordinates as gc
import gala.dynamics as gd
from gala.dynamics import mockstream
import rotation_matrix_helper as rmh
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic
import streamframe as sf
from gala.coordinates import reflex_correct

# Galactocentric reference frame to use for this project
galactocentric_frame = coord.Galactocentric(z_sun=0.*u.pc,
                                            galcen_distance=8.3*u.kpc)
vcirc = 238.*u.km/u.s
vlsr = [-11.1, 12.24, 7.25]*u.km/u.s

galcen_frame = dict()
galcen_frame['galactocentric_frame'] = galactocentric_frame
galcen_frame['vcirc'] = vcirc
galcen_frame['vlsr'] = vlsr


def integrate_forward_backward(potential, w0, t_forw, t_back, dt=0.5,
                               Integrator=gi.DOPRI853Integrator, t0=0.):
    """
    Integrate an orbit forward and backward from a point and combine
    into a single orbit object.

    Parameters
    ----------
    potential : :class:`gary.potential.PotentialBase`
    w0 : :class:`gary.dynamics.CartesianPhaseSpacePosition`, array_like
    t_forw : numeric
        The amount of time to integate forward in time (a positive number).
    t_back : numeric
        The amount of time to integate backwards in time (a negative number).
    dt : numeric (optional)
        The timestep.
    Integrator : :class:`gary.integrate.Integrator` (optional)
        The integrator class to use.
    t0 : numeric (optional)
        The initial time.

    Returns
    -------
    orbit : :class:`gary.dynamics.CartesianOrbit`
    """
    if t_back != 0:
        o1 = potential.integrate_orbit(w0, dt=-dt, t1=t0, t2=t_back, Integrator=Integrator)
    else:
        o1 = None

    if t_forw != 0:
        o2 = potential.integrate_orbit(w0, dt=dt, t1=t0, t2=t_forw, Integrator=Integrator)
    else:
        o2 = None

    if o1 is None:
        return o2
    elif o2 is None:
        return o1

    o1 = o1[::-1]
    o2 = o2[1:]
    #print(o1.t, o2.t)
    orbit = combine([o1, o2], along_time_axis=True)

    if orbit.pos.shape[-1] == 1:
        return orbit[:,0]
    else:
        return orbit

def _unpack(p, freeze=None):
    """ Unpack a parameter vector """

    if freeze is None:
        freeze = dict()

    # these are for the initial conditions
    phi2,d,mul,mub,vr = p[:5]
    count_ix = 5

    # time to integrate forward and backward
    if 't_forw' not in freeze:
        t_forw = p[count_ix]
        count_ix += 1
    else:
        t_forw = freeze['t_forw']

    if 't_back' not in freeze:
        t_back = p[count_ix]
        count_ix += 1
    else:
        t_back = freeze['t_back']

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        phi2_sigma = p[count_ix]
        count_ix += 1
    else:
        phi2_sigma = freeze['phi2_sigma']

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        d_sigma = p[count_ix]
        count_ix += 1
    else:
        d_sigma = freeze['d_sigma']

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        vr_sigma = p[count_ix]
        count_ix += 1
    else:
        vr_sigma = freeze['vr_sigma']

    if 'hernquist_logm' not in freeze:
        hernquist_logm = p[count_ix]
        count_ix += 1
    else:
        hernquist_logm = freeze['hernquist_logm']

    return phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma,hernquist_logm

def ln_prior(p, data, err, R, Potential, true_potential, dt, freeze):
    """
    Evaluate the prior over stream orbit fit parameters.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    """

    # log prior value
    lp = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma,hernquist_logm = _unpack(p, freeze)

    # time to integrate forward and backward
    t_integ = np.abs(t_forw) + np.abs(t_back)
    if t_forw <= t_back:
        raise ValueError("Forward integration time less than or equal to "
                         "backwards integration time.")

    if (t_forw != 0 and t_forw < dt) or (t_back != 0 and t_back > -dt):
        return -np.inf

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        if phi2_sigma <= 0.:
            return -np.inf
        lp += -np.log(phi2_sigma)

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        if d_sigma <= 0.:
            return -np.inf
        lp += -np.log(d_sigma)

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        if vr_sigma <= 0.:
            return -np.inf
        lp += -np.log(vr_sigma)

    # strong prior on phi2
    if phi2 < -np.pi/2. or phi2 > np.pi/2:
        return -np.inf
    lp += norm.logpdf(phi2, loc=0., scale=phi2_sigma)

    # uniform prior on integration time
    ntimes = int(t_integ / dt) + 1
    if t_integ <= 2. or t_integ > 1000. or ntimes < 4:
        return -np.inf

    if hernquist_logm < 10.5 or hernquist_logm > 12.5:
        return -np.inf

    return lp

from gala.util import atleast_2d

def _mcmc_sample_to_coord(p, R):
    p = atleast_2d(p, insert_axis=-1) # note: from Gala, not Numpy
    rep = sf.StreamFrame(phi1=p[0]*0.*galactic.to_dict()['angle'],
                         phi2=p[0]*galactic.to_dict()['angle'], # this index looks weird but is right
                         distance=p[1]*galactic.to_dict()['length'],
                         M=R)
    gal = rep.transform_to(coord.Galactic)
    galnew = coord.Galactic(l = gal.l,
                            b = gal.b,
                            distance = gal.distance,
                            pm_l_cosb = p[2]*galactic.to_dict()['angle']/galactic.to_dict()['time'],
                            pm_b = p[3]*galactic.to_dict()['angle']/galactic.to_dict()['time'],
                            radial_velocity = p[4]*galactic.to_dict()['length']/galactic.to_dict()['time'])
    return galnew

def _mcmc_sample_to_w0(p, R):
    p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
    c = _mcmc_sample_to_coord(p, R)
    x0 = c.transform_to(galactocentric_frame).cartesian.xyz.decompose(galactic).value
    vx_0 = c.transform_to(galactocentric_frame).v_x.decompose(galactic).value
    vy_0 = c.transform_to(galactocentric_frame).v_y.decompose(galactic).value
    vz_0 = c.transform_to(galactocentric_frame).v_z.decompose(galactic).value
    v_0 = np.concatenate((vx_0, vy_0, vz_0))[:,np.newaxis]
    w0 = np.concatenate((x0, v_0))
    return w0

def ln_likelihood(p, data, err, R, Potential, true_potential, dt, freeze):
    """ Evaluate the stream orbit fit likelihood. """

    chi2 = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma,hernquist_logm = _unpack(p, freeze)

    w0 = _mcmc_sample_to_w0([phi2,d,mul,mub,vr], R)#[:,0]

    # HACK: a prior on velocities
    vmag2 = np.sum(w0[3:]**2)
    chi2 += -vmag2 / (0.15**2)

    # integrate the orbit
    potential = Potential(m=10**hernquist_logm, c=true_potential.parameters['c'], units=galactic)
    orbit = integrate_forward_backward(potential, w0, t_back=t_back, t_forw=t_forw)

    # rotate the model points to stream coordinates
    model_c = orbit.to_coord_frame(coord.Galactic, **galcen_frame)
    model_oph = model_c.transform_to(sf.StreamFrame(M=R))
#      = model_c.transform_to(Ophiuchus)

    # model stream points in ophiuchus coordinates
    model_phi1 = model_oph.phi1
    model_phi2 = model_oph.phi2.radian
    model_d = model_oph.distance.decompose(galactic).value
    model_mul = model_c.pm_l_cosb.decompose(galactic).value
    model_mub = model_c.pm_b.decompose(galactic).value
    model_vr = model_c.radial_velocity.decompose(galactic).value

    # for independent variable, use cos(phi)
    data_x = np.cos(data['phi1'])
    model_x = np.cos(model_phi1)
    # data_x = ophdata.coord_oph.phi1.wrap_at(180*u.deg).radian
    # model_x = model_phi1.wrap_at(180*u.deg).radian
    ix = np.argsort(model_x)

    # shortening for readability -- the data
    phi2 = data['phi2'].radian
    dist = data['distance'].decompose(galactic).value
    mul = data['mul'].decompose(galactic).value
    mub = data['mub'].decompose(galactic).value
    vr = data['vr'].decompose(galactic).value

    # define interpolating functions
    order = 3
    # bbox = [-np.pi, np.pi]
    bbox = [-1, 1]
    phi2_interp = InterpolatedUnivariateSpline(model_x[ix], model_phi2[ix], k=order, bbox=bbox) # change bbox to units of model_x
    d_interp = InterpolatedUnivariateSpline(model_x[ix], model_d[ix], k=order, bbox=bbox)
    mul_interp = InterpolatedUnivariateSpline(model_x[ix], model_mul[ix], k=order, bbox=bbox)
    mub_interp = InterpolatedUnivariateSpline(model_x[ix], model_mub[ix], k=order, bbox=bbox)
    vr_interp = InterpolatedUnivariateSpline(model_x[ix], model_vr[ix], k=order, bbox=bbox)

    chi2 += -(phi2_interp(data_x) - phi2)**2 / phi2_sigma**2 - 2*np.log(phi2_sigma)

    _err = err['distance'].decompose(galactic).value
    chi2 += -(d_interp(data_x) - dist)**2 / (_err**2 + d_sigma**2) - np.log(_err**2 + d_sigma**2)

    _err = err['mul'].decompose(galactic).value
    chi2 += -(mul_interp(data_x) - mul)**2 / (_err**2) - 2*np.log(_err)

    _err = err['mub'].decompose(galactic).value
    chi2 += -(mub_interp(data_x) - mub)**2 / (_err**2) - 2*np.log(_err)

    _err = err['vr'].decompose(galactic).value
    chi2 += -(vr_interp(data_x) - vr)**2 / (_err**2 + vr_sigma**2) - np.log(_err**2 + vr_sigma**2)

    # this is some kind of whack prior - don't integrate more than we have to
#     chi2 += -(model_phi1.radian.min() - data['phi1'].radian.min())**2 / (phi2_sigma**2)
#     chi2 += -(model_phi1.radian.max() - data['phi1'].radian.max())**2 / (phi2_sigma**2)

    return 0.5*chi2


def ln_posterior(p, *args, **kwargs):
    """
    Evaluate the stream orbit fit posterior probability.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    Returns
    -------
    lp : float
        The log of the posterior probability.

    """

    lp = ln_prior(p, *args, **kwargs)
    if not np.isfinite(lp):
        return -np.inf

    ll = ln_likelihood(p, *args, **kwargs)
    if not np.all(np.isfinite(ll)):
        return -np.inf
    #print(lp + ll.sum())
    return lp + ll.sum()


def reflex_uncorrect(coords, galactocentric_frame=None):
    """Correct the input Astropy coordinate object for solar reflex motion.
    The input coordinate instance must have distance and radial velocity information. If the radial velocity is not known, fill the
    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        The Astropy coordinate object with position and velocity information.
    galactocentric_frame : `~astropy.coordinates.Galactocentric` (optional)
        To change properties of the Galactocentric frame, like the height of the
        sun above the midplane, or the velocity of the sun in a Galactocentric
        intertial frame, set arguments of the
        `~astropy.coordinates.Galactocentric` object and pass in to this
        function with your coordinates.
    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        The coordinates in the same frame as input, but with solar motion
        removed.
    """
    c = coord.SkyCoord(coords)

    # If not specified, use the Astropy default Galactocentric frame
    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric()

    v_sun = galactocentric_frame.galcen_v_sun

    observed = c.transform_to(galactocentric_frame)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] - v_sun)
    fr = galactocentric_frame.realize_frame(rep).transform_to(c.frame)
    return coord.SkyCoord(fr)

def combine(args, along_time_axis=False):
    """
    Combine the input `Orbit` objects into a single object.

    The `Orbits` must all have the same potential and time array.

    Parameters
    ----------
    args : iterable
        Multiple instances of `Orbit` objects.
    along_time_axis : bool (optional)
        If True, will combine single orbits along the time axis.

    Returns
    -------
    obj : orbit_like
        A single objct with positions and velocities stacked along the last axis.

    """

    ndim = None
    time = None
    pot = None
    pos_unit = None
    vel_unit = None
    cls = None

    all_pos = []
    all_vel = []
    all_time = []
    for x in args:
        if ndim is None:
            ndim = x.ndim
            pos_unit = x.xyz.unit
            vel_unit = x.v_xyz.unit
            time = x.t
            if time is not None:
                t_unit = time.unit
            else:
                t_unit = None
            pot = x.potential
            cls = x.__class__
        else:
            if x.__class__.__name__ != cls.__name__:
                raise ValueError("All objects must have the same class.")

            if x.ndim != ndim:
                raise ValueError("All objects must have the same dimensionality.")

            if not along_time_axis:
                if time is not None:
                    if x.t is None or len(x.t) != len(time) or np.any(x.t.to(time.unit).value != time.value):
                        raise ValueError("All orbits must have the same time array.")

            if x.potential != pot:
                raise ValueError("All orbits must have the same Potential object.")

        pos = x.xyz
        if pos.ndim < 3:
            pos = pos[...,np.newaxis]

        vel = x.v_xyz
        if vel.ndim < 3:
            vel = vel[...,np.newaxis]

        all_pos.append(pos.to(pos_unit).value)
        all_vel.append(vel.to(vel_unit).value)
        if time is not None:
            all_time.append(x.t.to(t_unit).value)

    norbits = np.array([pos.shape[-1] for pos in all_pos] + [vel.shape[-1] for pos in all_vel])
    if along_time_axis:
        if np.all(norbits == norbits[0]):
            all_pos = np.hstack(all_pos)*pos_unit
            all_vel = np.hstack(all_vel)*vel_unit
            if len(all_time) > 0:
                all_time = np.concatenate(all_time)*t_unit
            else:
                all_time = None
        else:
            raise ValueError("To combine along time axis, all orbit objects must have "
                             "the same number of orbits.")
        if args[0].pos.ndim == 2:
            all_pos = all_pos[...,0]
            all_vel = all_vel[...,0]

    else:
        all_pos = np.dstack(all_pos)*pos_unit
        all_vel = np.dstack(all_vel)*vel_unit
        if len(all_time) > 0:
            all_time = all_time[0]*t_unit
        else:
            all_time = None

    return cls(pos=all_pos, vel=all_vel, t=all_time, potential=args[0].potential)
