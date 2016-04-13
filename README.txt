
MAIN CODE: BooleanBioNet-InfoDyn.py ==> need to work on it!


#################################################################################
## 0. Prepare network with proper input format (split by tab)
data/example/example-net-edges.dat
data/example/example-net-edges.dat

Format: 	0-1) NODE_FILE (two columns)
			node-name (or node-index)	threshold
		0-2) EDGE_FILE 
 			node-name1	node-name2	weight-edge-from-node1-to-node2
#################################################################################


#################################################################################
## 1. Code to read network structure from files in ## 0
code/input_net.py
#################################################################################


#################################################################################
## 2. Code for updating states of each network
code/updating_rule.py
#################################################################################


#################################################################################
## 3. Code for time evolution of each network
code/time_evol.py

Function:	3-1) transition_map: generate transition mapping between every network state ==> need to be added!
	
		3-2) time_series: generate time series data for informational measures 
		
#################################################################################


#################################################################################
## 4. Code to measure TE and AI for a given time series data (over ensemble or trajectory)
code/info_dyn.py ==> need to complete generalized version! 

Function:	4-1)compute_TE
		4-2)compute_AI
#################################################################################

