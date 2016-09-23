from PIL import Image

def main():
	img = Image.open('./Lena.png')
	row, col = img.size

	img_out = img.copy()
	pix_o = img.load()
	pix = img_out.load()

	for r in range(row):
		for c in range(col):
			pix[row-(r+1), col-(c+1)] = pix_o[r, c]

	img_out.save("ans2.png")


if __name__ == '__main__':
	main()