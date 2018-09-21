
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 2 - Network Connectivity
# 
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

# In[31]:

import networkx as nx

# This line must be commented out when submitting to the autograder
#!head email_network.txt


# ### Question 1
# 
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# 
# *This function should return a directed multigraph networkx graph.*

# In[32]:

def answer_one():

    G = nx.read_edgelist('email_network.txt', data=[('Time', int)], create_using=nx.MultiDiGraph())
    
    return G

#answer_one()


# ### Question 2
# 
# How many employees and emails are represented in the graph from Question 1?
# 
# *This function should return a tuple (#employees, #emails).*

# In[10]:

def answer_two():
        
    G = answer_one()
    return (len(G.nodes()), len(G.edges()))

answer_two()


# ### Question 3
# 
# * Part 1. Assume that information in this company can only be exchanged through email.
# 
#     When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# *This function should return a tuple of bools (part1, part2).*

# In[11]:

def answer_three():
        
    G = answer_one()
    
    return (nx.is_strongly_connected(G), nx.is_weakly_connected(G))


# ### Question 4
# 
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# 
# *This function should return an int.*

# In[16]:

def answer_four():
        
    G = answer_one()
    subgraphs = list(nx.weakly_connected_component_subgraphs(G))
    largest = max(subgraphs, key=len)
    return len(largest)


# ### Question 5
# 
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# 
# *This function should return an int*

# In[33]:

def answer_five():
        
    G = answer_one()
    subgraphs = list(nx.strongly_connected_component_subgraphs(G))
    largest = max(subgraphs, key=len)
    return len(largest)


# ### Question 6
# 
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.
# 
# *This function should return a networkx MultiDiGraph named G_sc.*

# In[34]:

def answer_six():
    G = answer_one()
    subgraphs = list(nx.strongly_connected_component_subgraphs(G))
    largest = max(subgraphs, key=len)
    return largest


# ### Question 7
# 
# What is the average distance between nodes in G_sc?
# 
# *This function should return a float.*

# In[35]:

def answer_seven():
    G_sc = answer_six()
    return nx.average_shortest_path_length(G_sc)


# ### Question 8
# 
# What is the largest possible distance between two employees in G_sc?
# 
# *This function should return an int.*

# In[36]:

def answer_eight():
        
    G_sc = answer_six()
    
    return nx.diameter(G_sc)


# ### Question 9
# 
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# 
# *This function should return a set of the node(s).*

# In[38]:

def answer_nine():
       
    G_sc = answer_six()
    diameter = answer_eight()
    ecc = nx.eccentricity(G_sc)
    nodes = set([u for u in ecc.keys() if ecc[u] == diameter])
    return nodes
#answer_nine()


# ### Question 10
# 
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# 
# *This function should return a set of the node(s).*

# In[ ]:

def answer_ten():
        
    G_sc = answer_six()
    radius = nx.radius(G_sc)
    ecc = nx.eccentricity(G_sc)
    nodes = set([u for u in ecc.keys() if ecc[u] == radius])
    return nodes


# ### Question 11
# 
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# 
# How many nodes are connected to this node?
# 
# 
# *This function should return a tuple (name of node, number of satisfied connected nodes).*

# In[62]:

def answer_eleven():
        
    G_sc = answer_six()
    diameter = nx.diameter(G_sc)
    all_nodes = G_sc.nodes()
    nodes = []
    for i in range(0,len(all_nodes)):
        tmp = [n for n in all_nodes if nx.shortest_path_length(G_sc, source=all_nodes[i], target=n) == diameter]
        if len(tmp) > 0:
            nodes = nodes + [(all_nodes[i], len(tmp))]

    node = max(nodes, key = lambda x: x[1])
    return node


# ### Question 12
# 
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)? 
# 
# *This function should return an integer.*

# In[71]:

def answer_twelve():
    G_sc = answer_six()
    center_nodes = nx.center(G_sc)
    source_node = answer_eleven()[0]
    return len(nx.minimum_node_cut(G_sc, s=center_nodes[0], t=source_node))


# ### Question 13
# 
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# 
# *This function should return a networkx Graph.*

# In[67]:

def answer_thirteen():
        
    G_sc = answer_six()
    
    return nx.Graph(G_sc)


# ### Question 14
# 
# What is the transitivity and average clustering coefficient of graph G_un?
# 
# *This function should return a tuple (transitivity, avg clustering).*

# In[ ]:

def answer_fourteen():
    
    G_un = answer_thirteen()
    
    return (nx.transitivity(G_un), nx.average_clustering(G_un))

