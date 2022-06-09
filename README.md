# Guide for using rotation extractor node.
- The node needs to upload some files and used them. The files are provided in this URL:

    https://owncloud.roboroyale.eu/s/vpN28Nw5Uy4omn9/download

- you can run script initilize_env.py in tools folder to automatically download the needed files and put them in right paths.

- codes inside src folder are rosnodes. codes inside tools folder are scripts that provide some usefull functionalities like drawing performance and converting rosbags to videos.

- For running the node just run roslaunch on the launch file in the launch file. You canchange the name of published and subscribed topics from launch file.