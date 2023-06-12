import numpy as np
import pandas as pd
import itertools

class player(object):
    """
    player (class)

    Class to represent a player in the game. Contains useful attributes to compute the PPCF for each player.

    Inputs:
    1) pid: player ID
    2) team: team ID
    3) teamname: team name
    4) max_speeds: dictionary of maximum speeds for each player
    5) max_accs: dictionary of maximum accelerations for each player
    6) params: dictionary of parameters for the PPCF model
    7) GKid: ID of the goalkeeper for the team
    """

    def __init__(self, pid , team , teamname, max_speeds , max_accs , params, GKid):
        self.id             = pid
        self.is_gk          = self.id == GKid
        self.team           = team
        self.teamname       = teamname
        self.playername     = "{}_{}_".format( teamname , pid )
        # self.vmax           = max_speeds[pid]
        # self.amax           = max_accs[pid]
        self.vmax           = params["vmax"]
        self.amax           = params["amax"]
        self.reaction_time  = params["reaction_time"]
        self.tti_sigma      = params["tti_sigma"]
        self.lambda_att     = params["lambda_att"]
        self.lambda_def     = params["lambda_gk"] if self.is_gk else params["lambda_def"]
        self.get_position(team)
        self.get_velocity(team)
        self.PPCF = 0.0 

    def get_position(self,team):
        self.position = np.array( [ team[self.playername+"x"] , team[self.playername+"y"] ] )
        self.inframe  = not np.any( np.isnan( self.position ) )

    def get_velocity(self,team):
        self.velocity   = np.array( [ team[self.playername+"vx"] , team[self.playername+"vy"] ] )
        self.speed      = np.linalg.norm( self.velocity )
        self.direction  = self.velocity / ( self.speed + 1e-8 )
        # Checks if the player is moving in a given frame (not sure if this is useful yet)
        self.moving     = self.speed > 0.05
        
    def time_to_intercept(self , r_final):
        # Compute time to intercept assuming that the player will continue moving at the same speed and direction
        r_reaction = self.position + self.reaction_time * self.velocity 
        # time from the place at the end of self.reaction_time to the final position, assuming max acceleration starting from current speed
        ttvmax = self.vmax / self.amax
        tpr = (self.speed + np.sqrt( self.speed**2 + 2 * self.amax * np.linalg.norm( r_final - r_reaction ) )) / self.amax
        # Check to make sure that we aren't accelerating to a point where we exceed vmax
        if ttvmax > tpr :
            self.tti = self.reaction_time + tpr 
        else :
            # Reach the maximum speed and then continue at that speed
            dist_at_ttvmax = self.amax * ttvmax**2 / 2
            self.tti = self.reaction_time + ttvmax + ( np.linalg.norm( r_final - r_reaction ) - dist_at_ttvmax ) / self.vmax
        return self.tti
    
    def prob_to_intercept(self , t_arrival ):
        f = 1/(1.0 + np.exp( -(t_arrival - self.tti)/(np.sqrt(3)*self.tti_sigma/np.pi) ) )
        return f 

def initialize_players( team , teamname , max_speeds , max_accs , params , GKid ): 
    """
    initialize_players (function)

    Function to initialize the player objects for a given team

    Inputs:
    1) team: team dataframe
    2) teamname: team name
    3) params: dictionary of parameters for the PPCF model
    4) GKid: ID of the goalkeeper for the team

    """

    unique_ids = np.unique( [c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
    players = [] 
    for pid in unique_ids :
        team_player = player( pid , team , teamname , max_speeds , max_accs , params , GKid )
        if team_player.inframe :
            players.append( team_player )
    return players 

def check_offsides( attacking_players , defending_players , ball_position , GK_numbers , tol=0.1 ):
    defending_GK_id = GK_numbers[1] if attacking_players[0].teamname == "Home" else GK_numbers[0]
    assert defending_GK_id in [ p.id for p in defending_players], "Defending goalkeeper not found in defending players list"
    defending_GK = [ p for p in defending_players if p.id == defending_GK_id ][0]
    defending_half = np,sign( defending_GK.position[0] )
    second_deepest_defender_x = sorted( [defending_half * p.position[0] for p in defending_players ], reverse=True )[1]
    offside_line = max( second_deepest_defender_x , defending_half*ball_position[0] , 0.0 ) + tol 
    valid_players = [ p for p in attacking_players if p.position[0]*defending_half <= offside_line ]
    return valid_players

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

def default_model_params( time_to_control_veto=3 ):

    params = {}

    # Player model parameters
    params["amax"] = 7. # maximum player acceleration m/s/s
    params["vmax"] = 5. # maximum player speed m/s
    params["reaction_time"] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params["tti_sigma"] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params["kappa_def"] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params["lambda_att"] = 4.3 # ball control parameter for attacking team
    params["lambda_def"] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params["lambda_gk"] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params["average_ball_speed"] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params["int_dt"] = 0.04 # integration timestep (dt)
    params["max_int_time"] = 10 # upper limit on integral time
    params["model_converge_tol"] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params["time_to_control_att"] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params["time_to_control_def"] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params

from time import sleep

def generate_PPCF( target_position , attacking_players , defending_players , ball_start_pos , params ):

    if ball_start_pos is None or np.any( np.isnan( ball_start_pos ) ) :
        ball_travel_time = 0.0 
    else :
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos ) / params['average_ball_speed']
    
    # Compute time to intercept for each player
    tau_min_att =  np.nanmin( [ p.time_to_intercept( target_position ) for p in attacking_players ] )
    tau_min_def =  np.nanmin( [ p.time_to_intercept( target_position ) for p in defending_players ] )

    # If the ball is in an "obvious" situation, we just assign the corresponding pitch control probabilities
    if tau_min_att - max( ball_travel_time , tau_min_def ) >= params["time_to_control_def"] :
        return 0.0 , 1.0
    elif tau_min_def - max( ball_travel_time , tau_min_att ) >= params["time_to_control_att"] :
        return 1.0 , 0.0 
    else : 
        attacking_players = [ p for p in attacking_players if p.time_to_intercept( target_position ) - tau_min_att < params["time_to_control_att"] ]
        defending_players = [ p for p in defending_players if p.time_to_intercept( target_position ) - tau_min_def < params["time_to_control_def"] ]

        dT_array = np.arange( ball_travel_time - params["int_dt"] , ball_travel_time + params["max_int_time"] , params["int_dt"] )

        PPCF_att = np.zeros_like( dT_array )
        PPCF_def = np.zeros_like( dT_array )

        ptot = 0.0
        i = 1 

        while 1 - ptot > params["model_converge_tol"] and i < dT_array.size :

            T = dT_array[i]

            for player in attacking_players :
                dPPCFdT = (1-PPCF_att[i-1]-PPCF_def[i-1])*player.prob_to_intercept( T ) * player.lambda_att
                # print(dPPCFdT)
                assert dPPCFdT >= 0.0 , "Invalid attacking player probability"
                if i == 1: 
                    player.PPCF = dPPCFdT * params["int_dt"]
                else :
                    player.PPCF += dPPCFdT * params["int_dt"]
                PPCF_att[i] += player.PPCF

            for player in defending_players :

                dPPCFdT = (1-PPCF_att[i-1]-PPCF_def[i-1])*player.prob_to_intercept( T ) * player.lambda_def
                assert dPPCFdT >= 0.0 , "Invalid defending player probability"
                if i == 1: 
                    player.PPCF = dPPCFdT * params["int_dt"]
                else :
                    player.PPCF += dPPCFdT * params["int_dt"]
                PPCF_def[i] += player.PPCF

            ptot = PPCF_att[i] + PPCF_def[i]
            
            i += 1
        
        # print(PPCF_att[i-1] , " " , PPCF_def[i-1] , " " , ptot)
        if i >= len( dT_array ) :
            print("Integration failed to converge: %1.3f" % ( ptot ) )
        # print("Chilling...")
        # sleep(10)
        return PPCF_att[i-1] , PPCF_def[i-1]

def pitch_control_for_frame( tracking_home , tracking_away , 
                                GK_numbers , 
                                max_speeds_home , max_speeds_away , 
                                max_accs_home , max_accs_away , 
                                params ,
                                field_dimen = (106.,68.), 
                                n_grid_cells_x = 50, 
                                offsides=False ):
    
    ball_start_pos = np.array([tracking_home['ball_x'],tracking_home['ball_y']])
    
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x)*dx - field_dimen[0]/2. + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy - field_dimen[1]/2. + dy/2.
    
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    attacking_players = initialize_players(tracking_home,'Home',max_speeds_home, max_accs_home,params,GK_numbers[0])
    defending_players = initialize_players(tracking_away,'Away',max_speeds_away, max_accs_away,params,GK_numbers[1])

    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, GK_numbers)
    # calculate pitch pitch control model at each location on the pitch
    
    for (i,j) in itertools.product( range(len(ygrid)) , range(len(xgrid)) ):
        target_position = np.array( [xgrid[j], ygrid[i]] )
        PPCFa[i,j],PPCFd[i,j] = generate_PPCF(target_position, attacking_players, defending_players, ball_start_pos, params)

    # for i in range( len(ygrid) ):
    #     for j in range( len(xgrid) ):
    #         target_position = np.array( [xgrid[j], ygrid[i]] )
    #         PPCFa[i,j],PPCFd[i,j] = generate_PPCF(target_position, attacking_players, defending_players, ball_start_pos, params)
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    print(checksum)
    assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    print(checksum , "\n")
    return PPCFa , PPCFd , xgrid , ygrid
