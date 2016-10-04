import argparse

def output_ans(args):

	with open(args.output_file, 'w') as f:
		f.write('id,value\n')
		with open(args.input_file, 'r') as fin:
			for i, line in enumerate(fin.readlines()):
				if i > 0:
					a = float(line.strip().split(',')[1])
					a = round(a)
					out = 'id_'+str(i-1)+','+str(a)+'\n'
					f.write(out)
		

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--output_file', default='./round.csv', type=str)
	parser.add_argument('--input_file', default='./output_f333_time.csv', type=str)
	args = parser.parse_args()

	output_ans(args)