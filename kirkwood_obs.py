# basic imports
import numpy as np
import pandas as pd

# date/time functionality
import datetime
from datetime import date

# color and visualization
from tabulate import tabulate
from colorama import Fore

# skyfield (for locations of solar system objects)
import skyfield
import skyfield
from skyfield.api import load
from astropy.table import QTable

# astropy (coordinates, units, date/time functionality)
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.time import TimeDelta

# astroplan
import astroplan
from astroplan import FixedTarget
from astroplan.utils import time_grid_from_range
from astroplan import Observer
from astroplan import (AltitudeConstraint, AirmassConstraint,
                       AtNightConstraint, MoonIlluminationConstraint, MoonSeparationConstraint)


def make_planet_table(date_str):
    '''
    Generates AstroPy Table with positions (RA, Dec) of solar system planets on the given date.
    
    Input: Desired date (str, formatted as "YYYY-MM-DD")
    Output: Table of RA/dec for each object (AstroPy Table)
    '''

    y,m,d = [int(num) for num in date_str.split("-")] # split input date into constituents

    ts = load.timescale()
    t = ts.utc(y, m, d)
    planets = load('de421.bsp') # load ephemerides

    earth = planets["earth"] # establish earth location
    barycentric = earth.at(t)

    # other 7 planets
    pl_l = ["mercury", "venus", "mars", "jupiter barycenter", "saturn barycenter", "uranus barycenter", "neptune barycenter"]
    pl_names = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    ra_l = []
    dec_l = []

    # grab position of every other planet
    for pl in pl_l:
        planet = planets[pl]
        astrometric = barycentric.observe(planet)
        ra, dec, distance = astrometric.radec(t)
        ra, dec = float(ra._degrees), float(dec._degrees) # convert RA and dec to degrees
        ra_l.append(ra), dec_l.append(dec)
        
    t = QTable([pl_names, ra_l, dec_l]) # generate table
    return t

def make_constraints(alt_lim = (10, 80), moon_sep = 5, max_airmass = None, night_type = None, moon_illum = None):
    '''
    Creates list of desired observational constraints for use in sim_kirkwood_planets. Intended only as a helper function.
    
    Inputs (to be called in sim_kirkwood_planets, if needed)
    -----------------------------
    alt_lim (tuple): Lower and upper bounds (in deg) on allowable altitude of telescope, default is (10, 80)
    moon_sep (float): Minimum angular separation (in deg) from moon, default is 5
    max_airmass (float, optional): Maximum allowable airmass
    night_type (str, optional): Defines beginning and end of night, options are "civil", "naut", and "astro"
                        respectively for "civilian", "nautical", and "astronomical" definitions of twilight
    moon_illum (str, optional): Allowable moon phase, options are "grey" and "dark"
    
    
    Output
    ---------------------------
    List of observational constraints (List of Astroplan objects)
    '''
    
    constraints = [AltitudeConstraint(alt_lim[0]*u.deg, alt_lim[1]*u.deg), MoonSeparationConstraint(moon_sep*u.deg)]
    if max_airmass:
        constraints.append(AirmassConstraint(max_airmass))
    if night_type == "civ":
        constraints.append(AtNightConstraint.twilight_civil())
    if night_type == "naut":
        constraints.append(AtNightConstraint.twilight_nautical())
    if night_type == "astro":
        constraints.append(AtNightConstraint.twilight_astronomical())
    if night_type == "grey":
        constraints.append(MoonIlluminationConstraint.grey())
    if night_type == "dark":
        constraints.append(MoonIlluminationConstraint.dark()) 
        
    return constraints

def make_obs_grid(kirkwood, constraints, targets, t1_ust, t2_ust, dt = 0.5):
    '''
    Creates grids (NumPy arrays) defining observational efficacy for each object during the desired night.
    Intended only as a helper function for sim_kirkwood_planets.
    
    Inputs
    -----------------------------
    kirkwood (Astroplan Observer object): Object defining location of Kirkwood observatory.
                                            Passed automatically from sim_kirkwood_planets, does not require user alteration.
    
    constraints(list of Astroplan Constraint objects): Output from make_constraints. Passed automatically from
                                                        sim_kirkwood_planets, does not require user alteration.
    
    targets (list of Astroplan Target objects): List of objects definiting positions of target objects.
                                                Passed automatically from sim_kirkwood_planets, does not require user alteration.
    
    t1_ust, t2_ust (AstroPy.Time objects): Starting and ending time of observations in Universal Standard Time.
                                            Passed autmoatically form sim_kirkwood_planets, does not require user alteration.
    
    dt (float): Time interval of output observing schedule (in hours), default is 0.5 (30 mins). 
                Can be called directly in sim_kirkwood_planets, if needed.)
    
    Output
    ------------------------------
    time_grid (NumPy array): time array for observing schedule (in UST) expressed as Julian Date
    
    
    '''
    
    dt *= u.hour # add astropy units to dt
    grid_list = []
    time_grid = time_grid_from_range([t1_ust, t2_ust], # create time grid
                                 time_resolution=dt)
    for target in targets: # iterate over targets
        # initialize blank grid with size len(constraints)*len(time_grid)
        # each row is one of the requested constraints, each column is an interval of size dt during the observing run
        observability_grid = np.zeros((len(constraints), len(time_grid)))
        for i, constraint in enumerate(constraints):
            # populate observability grid
            # array value is 0 if the constraint is not met, 1 if satisfied
            observability_grid[i, :] = constraint(kirkwood, target, times=time_grid)
        grid_list.append(observability_grid) # make list of grids, one for each object
    big_grid = np.array(grid_list)
    # combine all constraints to make observing schedule
    # total_obs is an array of size num_objects*len(time_grid)
    # array value is 0 if object is not observable during that block, 1 if it is
    total_obs = np.prod(big_grid, axis = 1)
    # for each object calculate percentage of night it will be observable
    obs_percent = np.sum(total_obs, axis = 1)/total_obs.shape[1]
    return time_grid, total_obs, obs_percent


def sim_kirkwood_planets(date = str(date.today()), start_time = str(datetime.datetime.now().time()), duration = 4,
                        alt_lim = (10, 80), moon_sep = 5, max_airmass = None, night_type = None, moon_illum = None, dt = 0.5):
    '''
    Given date, time, and duration of an observing run (with optional observational constraints),
    returns table of object positions ranked by duration of observability, as well as rough observing schedule 
    detailing blocks of time when each object will be observable during the run.
    
    Required Inputs:
    -----------------------------
    date (str): desired date of observation, formatted as "YYYY-MM-DD", default is present day
    start_time (str): desired start time of observing run (in local time zone), formatted as "HH:MM", defaults to current clock time
    duration (float): approximate duration of observing run (in hours), defaults to 4
    
                
    Optional (Keyword) Inputs:
    -----------------------------
    All constraint keyword arguments in make_constraints, as well as the dt argument for make_obs_grid.
    
    Output
    ------------------------------
    Prints table of objects, ranked by duration of observability, with positions in RA/dec and alt/az
    Prints rough observing schedule detailing blocks of observing run when each object is observable
    (i.e. satisfies all imposed observational constraints)
    
    
    '''
    #print(date, start_time)
    t1 = Time(date + " " + start_time) # convert start time to time object
    t2 = t1 + TimeDelta(duration*u.h) # make end time and add units
    
    # define Observer object, input location of Kirkwood
    kirkwood = Observer(longitude=-86.5264*u.deg, latitude=39.1653*u.deg,
                  elevation=235*u.m, name="Kirkwood", timezone="US/East-Indiana")
    
    # convert EST to UST
    t1_ust, t2_ust = t1 + TimeDelta(4*u.h), t2 + TimeDelta(4*u.h)
    time_range = Time([t1_ust, t2_ust])

    # make table of objects
    target_table = make_planet_table(date)
    targets = [FixedTarget(coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), name=name) for name, ra, dec in target_table]
    
    # make constraints
    constraints = make_constraints(alt_lim, moon_sep, max_airmass, night_type, moon_illum)
    # make time grid and raw observing tables/schedule
    time_grid, total_obs, obs_percent = make_obs_grid(kirkwood, constraints, targets, t1_ust, t2_ust, dt)
    
    # convert ra and dec of each object to alt and az using Kirkwood's location
    alt, az = [], []
    for target in targets:
        alt.append(kirkwood.altaz(t1_ust, target).alt.value)
        az.append(kirkwood.altaz(t1_ust, target).az.value)
    
    # make lists for object names, ra, and dec
    pl_names, ra, dec = list(target_table['col0']), list(target_table['col1']), list(target_table['col2'])
    
    # make dataframe with object names and locations
    d = {'Object': pl_names,
        'Obs. Frac. of Night': obs_percent,
        'RA [deg]': ra,
        'Dec [deg]': dec,
        'Alt [deg]':alt,
        'Az [deg]':az
        }
    info_df = pd.DataFrame(d)
    info_df = info_df.set_index("Object")
    info_df = info_df.sort_values(by=['Obs. Frac. of Night'], ascending = False) # sort table by observability
    info_df = info_df.round(2)
  
    # display table
    print(tabulate(info_df, headers = 'keys', tablefmt = 'psql'))
    
    # convert time array back to EST
    time_est = [t.datetime - datetime.timedelta(hours=4) for t in time_grid]
    time_labels = [t.strftime("%H:%M") for t in time_est]
    
    # make dataframe for observing schedule
    time_df = pd.DataFrame(total_obs, columns = time_labels)
    time_df = time_df.replace([0, 1], [Fore.RED + "no"  + Fore.RESET, Fore.GREEN + 'YES' + Fore.RESET]) # add color
    time_df.insert(0, "Object", pl_names)
    time_df = time_df.set_index("Object")
    time_df = time_df.reindex(info_df.index) # sort objects to be in same order as other table
    # display schedule
    print(tabulate(time_df, headers = 'keys', tablefmt = 'psql'))
    
    return info_df, time_df

def_l = [str(date.today()), str(datetime.datetime.now().time()), 4, 10, 80, 5, None, None, None, 0.5]

message_list = ["Which date would you like to observe on? (Enter as YYYY-MM-DD, default is today):",
                "What time would you like to start your run? (Enter in military time as HH:MM, default is current local time):",
                "How many hours would you like to observe for? (default is 4)",
                "Would you like to change Kirkwood's minimum altitude limit (in degrees)? \n (type value if desired, otherwise hit ENTER. Default is 10.)",
                "Would you like to change Kirkwood's maximum altitude limit (in degrees)? \n (type value if desired, otherwise hit ENTER. Default is 80.)",
                "Would you like to change the minimum mandated distance (in degrees) from the Moon? \n (type value if desired, otherwise hit ENTER. Default is 5.)",
                "Would you like to specify a maximum allowed airmass? \n (type value if desired, otherwise hit ENTER. Default is None.)",
                "Would you like to specify a desired level of darkness for the night? \n (If desired, type 'civ' or 'naut' or 'astro' to respectively \n signify civil, nautical, or astronomical twilight. \n, otherwise hit ENTER. Default is None.)",
                "Would you like to specify a maximum brightness level for the moon? \n (If desired, type 'grey' or 'dark', otherwise hit ENTER. Default is None.)",
                "How often do you want to re-check for observability (in hours)? (type value if desired, otherwise hit ENTER. Default is 0.5, which creates a schedule in 30 minute intervals.)"]
                
# print interactive prompts in terminal
print("KIRKWOOD OBSERVING PLANNER")
print("CREATED BY: ARMAAN GOYAL, BRANDON RADZOM, JESSICA RANSHAW, XIAN-YU WANG")
print("ACCESSED ON %s"%str(datetime.datetime.now()))
print("------------------------------------------------------------------------")

for i in range(len(def_l)):
    print(message_list[i])
    var = input()
    if var != "":
        if i in [2, 3, 4, 5, 6, 9]:
            var = float(var)
        def_l[i] = var

print("Generating Schedule...")
sim_kirkwood_planets(date = def_l[0], start_time = def_l[1], duration = def_l[2],
                        alt_lim = (def_l[3], def_l[4]), moon_sep = def_l[5], max_airmass = def_l[6], night_type = def_l[7], moon_illum = def_l[8], dt = def_l[9])