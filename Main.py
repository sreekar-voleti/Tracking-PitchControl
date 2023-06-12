import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataCleaning as dc
import PitchControl as pc
import Visualization as vis

## Options 

path_to_tracking_data = '../sample-data/data'

match_name = 'Sample_Game_1'
frame = 832
field_dimen = (106.0,68.0)
include_player_velocities = True
n_x_grid_cells = 200
show = False  

## Code 

# Read in the tracking data
data_home = dc.read_tracking_data(path_to_tracking_data, match_name, "Home" )
data_away = dc.read_tracking_data(path_to_tracking_data, match_name, "Away" )

# Convert positions from metrica units to meters
data_home = dc.convert_units(data_home)
data_away = dc.convert_units(data_away)

# Reverse direction of play in the second half so that home team is always attacking from right->left
data_home = dc.standard_playing_direction( data_home )
data_away = dc.standard_playing_direction( data_away ) 

# Calculate player velocities and accelerations 
data_home = dc.compute_vel_and_acc( data_home )
data_away = dc.compute_vel_and_acc( data_away )

# Calculate max player speeds and accelerations 
max_speed_home , max_acc_home = dc.get_max_speeds_and_acc( data_home , "Home" )
max_speed_away , max_acc_away = dc.get_max_speeds_and_acc( data_away , "Away" )

PPCFa , PPCFd , xgrid , ygrid = pc.pitch_control_for_frame( data_home.loc[frame] , data_away.loc[frame] ,
                                                    [1,1] ,  
                                                    max_speed_home , max_speed_away , 
                                                    max_acc_home , max_acc_away , 
                                                    pc.default_model_params() ,
                                                    n_grid_cells_x=n_x_grid_cells)
    
fig,ax = vis.plot_pitch( field_color='white', field_dimen = field_dimen)
vis.plot_frame( data_home.loc[frame], data_away.loc[frame] , figax=(fig,ax) , PlayerAlpha=0.5, include_player_velocities=include_player_velocities, annotate=True)
cmap = 'seismic'
ax.imshow( np.flipud(PPCFa) , extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.) ,cmap=cmap,alpha=0.5 )
plt.savefig("Figures/PitchControlExample.png")
if show == True:
    plt.show()
