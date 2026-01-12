from common.lib.CamConfig import CamConfig 
from common.lib.Processing import * 

Video_path = "Modularize/testVid.mp4"
outPath = "Modularize/test.json"

gcps = dict(
    src=[ # Values to be switched with variables that will get the values from the website
        [1780 , 900],   #Top right
        [1250 , 660],   #Bottom right 
        [500, 660],     #Bottom left 
        [30, 900]       #Tope left
    ],
    dst = [
    [642735.8076, 8304292.1190],  # lowest right coordinate
    [642737.5823, 8304295.593],  # highest right coordinate
    [642732.7864, 8304298.4250],  # highest left coordinate
    [642732.6705, 8304296.8580]  # highest right coordinate
    ],
    z_0 = 1182.2 
)

corners = [ # x,Y coordinates (column,row)
    [292, 817],     # top left 
    [50, 166],      # btm left
    [1200, 236],    # btm right 
    [1600, 834]     # top right
]

cfg = CamConfig(Video_path , gcps , corners , outPath)

bbox_coords = [   # Approx. border of water , User ip from the website 
    [780, 600],     # btm left 
    [0, 900],       # top left
    [1800, 900],    # top right
    [1200, 600]     #btm right
]

NCPath = "Modularize/test.nc"
piv = process(Video_path , outPath , bbox_coords , NCPath)


maskPath = "Modularize/test_mask.nc"
mask_piv = mask(Video_path , NCPath , maskPath)


