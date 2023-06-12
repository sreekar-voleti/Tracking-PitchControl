import numpy as np
import scipy.signal as signal
import pandas as pd
import csv as csv

def read_tracking_data( path_to_tracking_data , match_name , team_name ):
    """
    read_tracking_data (function)

    Function to clean up tracking data.
    
    Input:
    1) path_to_tracking_data: path to the folder containing the tracking data
    2) match_name: the name of the match, e.g. 'Sample_Game_1'
    3) team_name: name of the team in the filename of the tracking data, e.g. 'Home' or 'Away'

    Output:
    1) tracking: a Pandas dataframe containing the tracking data for the chosen match and team

    """

    # Get the name of the file and open the csv file
    match_filename = "{}/{}/{}_RawTrackingData_{}_Team.csv".format( path_to_tracking_data , match_name , match_name , team_name )
    csvfile = open( match_filename , 'r' )
    reader = csv.reader( csvfile )
    next(reader)
    print( "Reading team: " , team_name )

    # Get jersey numbers for the renaming of player position columns
    jerseys = [ x for x in next( reader ) if x != '' ]
    columns = next( reader ) # This row contains the current column headers

    # Change names of player position columns. The Time[s], Period and Frame columns remain unchanged.
    for i,j in enumerate( jerseys ):
        columns[ i * 2 + 3 ] = "{}_{}_x".format( team_name , j )
        columns[ i * 2 + 4 ] = "{}_{}_y".format( team_name , j )

    # The ball columns are also unchanged 
    columns[ -2 ] = "ball_x"
    columns[ -1 ] = "ball_y"

    # Now read in all the values from the csv files into the newly named columns
    tracking = pd.read_csv( match_filename , names=columns, index_col='Frame', skiprows=3)
    return tracking

def convert_units( data , field_dimen=[106.,80.] ):
    """
    convert_units (function)

    Metrica defines their coordinates from 0 to 1 so we need to get it into some real world units.
    Function to convert the coordinates of the tracking data from Metrica units to meters, with the origin at the center circle.

    Inputs:
    1) data: tracking data dataframe
    2) field_dimen: dimensions of the field in meters, default to Metrica dimensions (106,68)

    Outputs:
    1) data: tracking data dataframe with the coordinates converted to meters
    """

    x_cols = [ c for c in data.columns if c[-1].lower() == 'x' ]
    y_cols = [ c for c in data.columns if c[-1].lower() == 'y'  ]

    # We want to make the origin the center of the pitch, so we need to subtract 0.5 from the x and y coordinates.
    data[x_cols] = ( data[x_cols] - 0.5 ) * field_dimen[0]
    # They define their origin as the top left corner, so we need to flip the y coordinates.
    data[y_cols] = -1 * ( data[y_cols] - 0.5 ) * field_dimen[1]

    return data

def standard_playing_direction( data ):
    """
    standard_playing_direction (function)

    Function to standardize the direction of play for a single period.

    Inputs:
    1) data: tracking data dataframe

    Outputs:
    1) data: tracking data dataframe with the direction of play standardized
    """
    # When we are in period 2, invert the x and y coordinates
    columns = [ c for c in data.columns if c[-1].lower() in ['x','y'] ]
    data.loc[ data.Period==2 , columns ] *= -1
    return data 

def compute_vel_and_acc( data , 
                        smoothing = True ,
                        max_speed = 14 , # [m/s] Slightly faster than Usain Bolt's fastest ever recorded time 
                        max_acceleration = 9.5 , # [m/s^2] Also from Usain Bolt's record run,
                        filter = "Savitzky-Golay" , 
                        window_size = 7 ,
                        polyorder = 1 ):
    
    """
    compute_vel_and_acc (function)

    Function to compute velocities and accelerations for each player and the ball.

    Inputs:
    1) data: tracking data dataframe
    2) smoothing: boolean variable to determine whether to smooth the velocities and accelerations
    3) max_speed: maximum speed allowed for a player, in meters per second
    4) max_acceleration: maximum acceleration allowed for a player, in meters per second squared
    5) filter_: type of filter to use for smoothing, either 'Savitzky-Golay' or 'Moving Average'
    6) window_size: size of the window for the smoothing filter
    7) polyorder: order of the polynomial for the Savitzky-Golay filter

    Outputs:
    1) data: tracking data dataframe with velocities and accelerations added
    """

    # Remove existing velocities and accelerations. 
    remove_player_vel_and_acc( data )

    # Unique player IDs
    player_ids = np.unique( [ c[:-2] for c in data.columns[2:] ] )

    # Time steps 
    dt = data["Time [s]"].diff()

    # Get player and ball columns
    x_cols = [ c for c in data.columns if c[-1].lower() == 'x' ]
    y_cols = [ c for c in data.columns if c[-1].lower() == 'y' ]

    for player in player_ids:
        # Get player x and y positions

        print( "Calculating velocities and accelerations for player: " , player)

        vx = data["{}_x".format(player)].diff() / dt
        vy = data["{}_y".format(player)].diff() / dt

        # Get rid of unphysical velocities (Ball does not have a max speed in this approximation)
        if player != "ball" and max_speed > 0 :
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed > max_speed ] = np.nan
            vy[ raw_speed > max_speed ] = np.nan

        # Smooth velocities
        if smoothing : 
            if filter == "Savitzky-Golay" :
                vx = signal.savgol_filter( vx , window_length=window_size , polyorder=polyorder )
                vy = signal.savgol_filter( vy , window_length=window_size , polyorder=polyorder )
            elif filter == "moving average" :
                vx = vx.rolling( window_size , center=True ).mean()
                vy = vy.rolling( window_size , center=True ).mean()

        data["{}_vx".format(player)] = vx
        data["{}_vy".format(player)] = vy
        data["{}_speed".format(player)] = np.sqrt( vx**2 + vy**2 )

        ax = data["{}_vx".format(player)].diff() / dt
        ay = data["{}_vy".format(player)].diff() / dt

        if player != "ball" and max_speed > 0 :
            raw_acc = np.sqrt( ax**2 + ay**2 )
            ax[ raw_acc > max_acceleration ] = np.nan
            ay[ raw_acc > max_acceleration ] = np.nan

        if smoothing : 
            if filter == "Savitzky-Golay" :
                ax = signal.savgol_filter( ax , window_length=window_size , polyorder=polyorder )
                ay = signal.savgol_filter( ay , window_length=window_size , polyorder=polyorder )
            elif filter == "Moving_Average" :
                ax = vx.rolling( window_size , center=True ).mean()
                ay = vy.rolling( window_size , center=True ).mean()

        data["{}_ax".format(player)] = ax
        data["{}_ay".format(player)] = ay
        data["{}_acc".format(player)] = np.sqrt( ax**2 + ay**2 )

    return data 

def remove_player_vel_and_acc(data):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in data.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acc']] # Get the player ids
    data = data.drop(columns=columns)
    return data

def get_max_speeds_and_acc( team , teamname ):
    """
    get_max_speeds_and_acc (function)

    Provide a dict with each player's max acceleration and speed. 
    This is actually a bad way to do this, since it needs you to iterate over the whole team dataframe (look into the future).
    Ideally, this would be some property of the player based on past games, but for now it is a placeholder. 

    Inputs:
    1) team: team dataframe

    Outputs:
    1) max_speeds: dictionary of max speeds for each player
    2) max_accs: dictionary of max accelerations for each player
    """   
    player_ids = np.unique( [ c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
    max_speeds = {}
    max_accs = {}
    for pid in player_ids:
        playername      = "{}_{}_".format( teamname , pid )
        max_speeds[pid] = max( team[playername+"speed"].dropna() )
        max_accs[pid]   = max( team[playername+"acc"].dropna() )
    return max_speeds , max_accs

def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    ''' 
    x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in ['Home','Away']]
    GK_col = team.iloc[0][x_columns].abs().idxmax(axis=1)
    return GK_col.split('_')[1]