import numpy as np 
import sys

def main():

	col_index = int(sys.argv[1])
	file_name = sys.argv[2]

	mat = []

	with open(file_name, 'r') as f:
		for line in f.readlines():
			line = line.strip().split(' ')
			tmp_row = map(float, line)
			mat.append(tmp_row)

	numpy_mat = np.matrix(mat)

	out = ','.join([str(float(i)) for i in sorted(numpy_mat[:, col_index])])
	print out




if __name__ == '__main__':
	main()