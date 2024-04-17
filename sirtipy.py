# SiRTIPY: Simple Radiative Transfer In Python
# by Jeremy Bailin
# December 2015

# Do simple 1D integration of the radiative transfer equation given emission and
# absorption coefficients as a function of location and frequency.

import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt

class region(object):
    """Object that defines the emission and absorption within a region."""
    def __init__(self, emission=False, emission_args=False,
            absorption=False, absorption_args=False):
        """Creates the region object and optionally initializes it with emission
        and/or absorption functions.

        Optional arguments:
          emission: A function that returns the emission. See the
                    add_emission_function description for details.
          emission_args: A tuple with any additional arguments that need to be
                    passed to the emission function.
          absorption: A function that returns the absorption. See the
                    add_absorption_function description for details.
          absorption_args: A tuple with any additional arguments that need to be
                    passed to the absorption function."""

        # Add the emission and absorption functions to a list. If ..._args are also
        # given, then those will be fed in as additional parameters to the given
        # function.
        self.emission_funcs = []
        if emission:
            self.add_emission_func(emission, emission_args)
        self.absorption_funcs = []
        if absorption:
            self.add_absorption_func(absorption, absorption_args)

    def add_emission_func(self, func, args):
        """Use this to add an emission process to the region. func is the
        reference to the function that specifies the amount of emission,
        and args is a tuple containing any additional arguments that must be passed to func.
        Multiple emission functions may be added for a region if multiple
        processes are important.
        
        The emission function must take at least three parameters:
        
           1. Frequency (in units of Hz). This should be a numpy array.
           2. 1D coordinate (in units of cm). This should be a float.
           3. A spectrum object that gives the current I_nu spectrum. This is
              necessary for some scattering processes.
        
        Additional arguments can be given as a tuple in args.
        The function should return the value of the emission
        coefficient j_nu, as defined in R+L section 1.4, in cgs units (i.e.
        erg/s/cm3/Hz/steradian).
       
        For example, the following function adds the emission for a gas cloud
        running from 0 to 1e21 cm, where the emission is a constant value of
        1e-20 erg/s/cm3/Hz/ster (the value is given as an additional argument).

        def j_constant(nu, s, Inu, jvalue):
            if (s>0.) and (s<1e21):
                return jvalue
            else
                return 0.

        new_region = sirtipy.region()
        new_region.add_emission_func(j_constant, (1e-20,))"""

        self.emission_funcs.append( (func, args,) )

    def add_absorption_func(self, func, args):
        """Use this to add an absorption process to the region. func is the
        reference to the function that specifies the amount of absorption, and
        args is a tuple containing any additional arguments that must be passed to func.
        Multiple absorption functions may be added for a region if multiple
        processes are important.
        
        The absorption function must take at least three parameters:
        
           1. Frequency (in units of Hz). This should be a numpy array.
           2. 1D coordinate (in units of cm). This should be a float.
           3. A spectrum object that gives the current I_nu spectrum. This is
              necessary for some scattering processes.
        
        Additional arguments can be given as a tuple in args.
        The function should return the value of the absorption
        coefficient alpha_nu, as defined in R+L section 1.4, in cgs units (i.e.
        cm^-1).
       
        For example, the following function adds the absorption for a gas cloud
        running from 0 to 1e21 cm, where the absorption value is alpha_1 at
        frequencies below nu_0, and alpha_2 at frequencies above nu_0, with
        alpha_1, alpha_2, and nu_0 given as extra arguments with values of 1e-9,
        1e-11, and 1e14 respectively.

        def alpha_stepfunc(nu, s, Inu, alpha1, alpha2, nu0):
            if (s>0.) and (s<1e21):
                return (nu < nu0)*alpha1 + (nu >= nu0)*alpha2
            else:
                return 0.

        new_region = sirtipy.region()
        new_region.add_absorption_func(alpha_stepfunc, (1e-9, 1e-11, 1e14,))"""

        self.absorption_funcs.append( (func, args,) )

    def j_nu(self, location, frequency, Inu):
        """Returns the value of the total emission coefficient in the region at
        the given location at the given frequency, given the current radiation
        field Inu."""
        emission = 0.
        # Run through all defined functions and add all of their effects.
        for func, args in self.emission_funcs:
            emission += func(frequency, location, Inu, *args)

        return emission

    def alpha_nu(self, location, frequency, Inu):
        """Returns the value of the total absorption coefficient in the region at
        the given location at the given frequency, given the current radiation
        field Inu."""
        absorption = 0.
        # Run through all defined functions and add all of their effects.
        for func, args in self.absorption_funcs:
            absorption += func(frequency, location, Inu, *args)

        return absorption


class spectral_array(object):
    """Object that contains a 1D array of spectrum objects, and can be used to
    return a 2D array of intensity vs. array location."""
    def __init__(self):
        """Creates an empty spectral_array object."""
        self.spectra = []

    def add_spectrum(self, newspec):
        """Add a new spectrum to the spectral array."""
        self.spectra.append(newspec)

    def __getitem__(self,key):
        """Return spectrum number key."""
        return self.spectra[key]

    def intensities(self):
        """Return a 2D array (nlocations, nfreq) containing the intensity of the
        spectra as a function of spectrum location index."""
        nspec = len(self.spectra)
        nfreq = len(self.spectra[0].axis.frequencies)
        intensity_2d = np.zeros((nspec, nfreq,))
        for i in range(nspec):
            intensity_2d[i,:] = self.spectra[i].intensity

        return intensity_2d


class spectrum(object):
    """Object that contains a frequency axis and associated intensities."""

    def __init__(self, spec='empty', frequencies=False, intensities=False, temperature=False):
        """Creates a spectrum object, consisting of a frequency axis and
        associated light intensity.
        
        The frequency axis is initialized with the freqencies keyword, which can
        either be a frequency_axis object, or an astropy Quantity
        array that contains the frequency (or wavelength) values.

        Intensities are initialized according to spec:
           'empty':         All intensities are zero.
           'verbatim':      intensities is a numpy array containing the
                            intensity values, in cgs units (erg/s/cm2/Hz/ster).
           'blackbody':     Initialize to a blackbody spectrum with the given
                            temperature, in Kelvin."""

        # Initialize frequency axis.
        # Check whether frequencies is already a frequency_axis object, or if we need to create
        # a new frequency_axis object from it.
        if isinstance(frequencies, frequency_axis):
            self.axis = frequencies
        else:
            self.axis = frequency_axis('verbatim', frequencies=frequencies)
        self.nspec = len(self.axis.frequencies)

        # Initialize intensity.
        if spec=='empty':
            self.intensity = np.zeros(self.nspec)
        if spec=='verbatim':
            self.intensity = np.array(intensities)
        if spec=='blackbody':
            self.intensity = blackbody_intensity(temperature, self.axis.frequencies)

    def plot(self, xunit=False, yunit=False, *args, **kwargs):
        """Plot the spectrum. This will create new axes if necessary, or
        overplot on existing axes if there are any (i.e. it uses
        matplotlib.pyplot.plot).

        The default unit for the x-axis is Hz, and the default y-axis unit is
        erg/s/cm2/Hz/ster. Both can be overridden using the xunit and yunit
        keywords respectively, which should be astropy Unit or Quantities. xunit
        can be a wavelength instead of a frequency.

        Additional keywords are passed through to matplotlib.pyplot.plot."""

        # Convert x and y values to desired units if necessary.
        nu_Hz = u.Quantity(self.axis.frequencies, u.Hz)
        intensity_cgs = u.Quantity(self.intensity, u.erg / u.s / u.cm**2 / u.Hz)
        if xunit:
            xaxis = nu_Hz.to(xunit, equivalencies=u.spectral())
        else:
            xaxis = nu_Hz
        if yunit:
            yaxis = intensity_cgs.to(yunit)
        else:
            yaxis = intensity_cgs


        # Plot spectrum.
        plt.plot(xaxis, yaxis, *args, **kwargs)

        # Label axes appropriately. Check if the xaxis unit is a length, in
        # which case it is a wavelength instead of a frequency.
        if xaxis.si.unit == u.m:
            plt.xlabel('Wavelength [%s]' % xaxis.unit.to_string(format='latex_inline'))
        else:
            plt.xlabel('Frequency [%s]' % xaxis.unit.to_string(format='latex_inline'))
        plt.ylabel('Intensity [%s]' % yaxis.unit.to_string(format='latex_inline'))



class frequency_axis(object):
    """Object containing a frequency axis for a spectrum."""

    def __init__(self, frequency_spec=False, frequencies=False, frange=False,
            numpts=100):
        """Create a frequency_axis object, and initialize it if frequency_spec
        is given. See add_frequency_region for details."""

        # Create a null array of frequencies, and add a spectral region if given.
        self.frequencies = np.array([])
        if frequency_spec:
            self.add_frequency_region(frequency_spec, frequencies=frequencies,
                    frange=frange, numpts=numpts)
            

    def add_frequency_region(self, frequency_spec, frequencies=False,
            frange=False, numpts=100):
        """Add a region of spectrum for which radiative transfer will be
        computed. Multiple regions can be added; for example, you could add a
        broad frequency region at low spectral resolution to determine the
        continuum, and then a region within it at high spectral resolution
        where there is a narrow feature like a spectral line.
        
        Possible values for frequency_spec:
           'verbatim': frequencies will contain an array of the exact
                       wavelengths or frequencies that should be added.
           'linear':   create a region with points spaced linearly in
                       frequency. frange will contain the endpoints of the
                       region and numpts will contain the number of sampling
                       points (default: 100).
           'log':      create a region with points spaced logarithmically in
                       frequency. frange will contain the endpoints of the
                       region and numpts will contain the number of sampling
                       points (default: 100).
        
        IMPORTANT: frequencies or frange must have astropy units! These may be
        wavelength units instead of frequencies."""

        # Internal unit is Hz
        frequnit = u.Hz

        # Just assign frequencies to input values if verbatim.
        if frequency_spec=='verbatim':
            new_freq = frequencies.to(frequnit, equivalencies=u.spectral()).value

        # If using a range, first convert to the internal unit.
        if frange:
            new_frange = [frange[0].to(frequnit, equivalencies=u.spectral()).value,
                    frange[1].to(frequnit, equivalencies=u.spectral()).value]
        # Linearly spaced frequencies using np.linspace.
        if frequency_spec=='linear':
            new_freq = np.linspace(new_frange[0], new_frange[1], num=numpts)
        # Logarithmically spaced frequencies using np.logspace. Note that
        # np.logspace expects log10 of the endpoints, not the endpoints
        # themselves.
        if frequency_spec=='log':
            new_frange = np.log10(new_frange)
            new_freq = np.logspace(new_frange[0], new_frange[1], num=numpts)

        # Sort the frequency axis in case new frequencies were added in the
        # middle of existing values.
        self.frequencies = np.sort(np.append(self.frequencies, new_freq))



def radiative_transfer(region, input_spectrum, locations, ds, printevery=False):
    """Perform radiative transfer from RL equation 1.23 using finite
    differences.

    Inputs:
        region:           A region object with emission and/or absorption functions defined.
        input_spectrum:   A spectrum object containing the initial conditions.
        locations:        A 2-element array containing the coordinate range
                          along which the radiative transfer will be computed (from locations[0]
                          to locations[1] -- it is ok if locations[0] > locations[1]). Must be
                          in cm and in the same coordinate system as used by any emission or
                          absorption functions defined in region.
        ds:               Finite difference step size, in cm. Should be tuned to
                          be quite a bit smaller than the scale on which the region changes and/or
                          the intensity changes along the ray.
        printevery:       Give an update on the progress of the calculation every printevery steps.

    Outputs a tuple (spectral_array, optical_depth, location_axis):
        spectral_array:   A 1D array of spectrum objects. spectral_array[0] is
                          the input spectrum, spectral_array[-1] is the final spectrum, and
                          spectral_array[i] is the spectrum along the ray at location_axis[i].
        optical_depth:    A 2D array (nlocations x nfreq) of optical depths tau as a function of
                          frequency along the ray. optical_depth[i,j] is the
                          optical depth at point location_axis[i], at frequency j.
        location_axis:    Location along the ray. location_axis[0] is
                          locations[0] and location_axis[-1] is locations[1]. Note that
                          location[i] might not be exactly locations[0]+i*ds because ds is
                          modified to give an integer number of steps."""

    # Flag to make sure we only give the "too large ds" warning once. If the input spectrum is initialized
    # to a real spectrum, then you should pay attention to the warning, but if
    # it is initalized to 'empty', you will almost certainly get a warning at
    # step 2 no matter what.
    warningflag = False

    # Modify ds so that the number of steps is an integer.
    distance = locations[1]-locations[0]
    nstep = int(np.abs(np.round(distance / ds)))
    ds = distance / nstep

    # Set up empty arrays to hold the results of the calculation.
    # spectran is the spectrum at each point along the integration.
    # tau is the optical depth spectrum at each point along the integration.
    # locarray is the location along the integration.
    spectran = spectral_array()
    tau = np.zeros( (nstep+1, input_spectrum.nspec,) )
    locarray = np.zeros(nstep+1)

    # Initialize with the input spectrum and the beginning location.
    spectran.add_spectrum(spectrum('verbatim',
        intensities=input_spectrum.intensity, frequencies=input_spectrum.axis))
    locarray[0] = locations[0]
    
    # Iterate through each location.
    for stepi in range(nstep):
        if printevery:
            # Progress update
            if (stepi+1) % printevery == 0:
                print('Location %d of %d' % (stepi+1, nstep))
        # Where in the region are we?
        s = stepi * ds + locations[0]

        # Evaluate the emission and absorption at the half-step, which improves
        # numerical convergence.
        s_half = s + 0.5*ds
        alpha = region.alpha_nu(s_half, spectran[stepi].axis.frequencies, spectran[stepi].intensity)
        j = region.j_nu(s_half, spectran[stepi].axis.frequencies, spectran[stepi].intensity)

        # Update spectrum and optical depth according to RL equations 1.23 and
        # 1.26 respectively.
        dI = j * ds - alpha * spectran[stepi].intensity * ds
        # if differential change is more than 10%, warn that ds might be too
        # large. Only makes sense if initial intensity is not zero.
        # Only warn once.
        if not warningflag:
            if (np.all(spectran[stepi].intensity != 0.) and (np.max(np.abs(dI / spectran[stepi].intensity)) > 0.1)):
                print('WARNING: Possible numerical convergence issue.')
                print(' Try again with a smaller value of ds to see if result changes.')
                warningflag = True
        spectran.add_spectrum(spectrum('verbatim', intensities=spectran[stepi].intensity + dI,
                frequencies=input_spectrum.axis))
        tau[stepi+1,:] = tau[stepi,:] + alpha * ds
        locarray[stepi+1] = s + ds

    # and return it!
    return (spectran, tau, locarray,)


def blackbody_intensity(temperature, freq):
    """Intensity B_nu of a blackbody of the given temperature, in cgs units. See RL equation 1.51."""
    # Note that this is B_nu, **not B_lambda**.
    h_over_csquared = (const.h / const.c**2).cgs.value
    h_over_k = (const.h / const.k_B).cgs.value
    return 2. * h_over_csquared * freq**3 / (np.exp(h_over_k * freq / temperature) - 1.)


def latex_float(f):
    """Stack Overflow solution to autoamtically formatting numbers for Latex."""
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


