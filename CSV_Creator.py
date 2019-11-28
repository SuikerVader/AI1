#Make csv labels for data 
import csv
import os
import glob
import ntpath

base_dir = 'Paintings'
rubens = 'Rubens'
brueghel = 'Brueghel'
picasso= 'Picasso'
mondriaan= 'Mondriaan'

Painters = ['Rubens','Brueghel','Picasso','Mondriaan']

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail 

for painter in Painters:
	csvData = []
	for line in glob.glob(base_dir+'/'+painter+'ALL/*'): #All is the directory name
		file = [path_leaf(line),painter]
		csvData.append(file)
	with open(painter+'.csv',mode = 'w') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(csvData)

