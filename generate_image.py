from PIL import Image, ImageFont, ImageDraw
import numpy as np


def generate_1lines_image(template, bg='background/bg1.jpg', size=(470, 110)):
	font = ImageFont.truetype("MyFont-Regular_ver3.otf", 116)
	font1 = ImageFont.truetype("Oswald-Regular.ttf", 112)
	old_style = False
	if type(bg).__name__ == 'str':
		im = Image.open(bg)
	else:
		im = bg
	if len(template) >= 12:
		im = im.resize((600, 110))
	elif len(template) >= 10:
		im = im.resize((540, 110))
	elif len(template) <= 8:
		old_style = True
		im = im.resize((480, 110))
	else:
		im = im.resize(size)
	width, height = im.size
	draw = ImageDraw.Draw(im)
	textsize = font.getsize(template)
	textX = int((width - textsize[0]) / 2-20)
	textY = int((height - textsize[1]) / 2)
	fill = tuple(np.random.randint(20, size=3))
	currX = textX
	for char in template:
		if char in ["M","N","H","A"]:
			if old_style:
				draw.text((currX, textY-40), char, font=font1, fill=fill)
				currX += font1.getsize(char)[0]
			else:
				draw.text((currX, textY), char, font=font, fill=fill)
				currX += font.getsize(char)[0]
		else:
			draw.text((currX, textY), char, font=font, fill=fill)
			currX  += font.getsize(char)[0]
	return im, textsize


def generate_2lines_images(template, bg='background/bg2.jpg', margin=12, size=(560, 400)):
	font1 = ImageFont.truetype("Oswald-Regular.ttf", 218)
	old_style = False
	font = ImageFont.truetype("MyFont-Regular_ver3.otf", 220)
	if type(bg).__name__ == 'str':
		im = Image.open(bg)
	else:
		im = bg

	if len(template) < 10:
		im = im.resize((450, 400))
		old_style = True
	else:
		im = im.resize(size)
	width, height = im.size
	draw = ImageDraw.Draw(im)
	line_1, line_2 = template.split('/')

	textsize1 = font.getsize(line_1)
	textX1 = int((width - textsize1[0]) / 2 - 20)
	textY1 = int((height / 2 - textsize1[1]) / 2) + margin
	textsize2 = font.getsize(line_2)
	textX2 = int((width - textsize2[0]) / 2)
	textY2 = int(height / 2 + (height / 2 - textsize2[1]) / 2) - margin / 2
	fill = tuple(np.random.randint(20, size=3))

	shadow = tuple(np.random.randint(200, 255, size=3))
	direction = tuple(np.random.randint(-3, 3, size=2))

	currX = textX1
	for char in line_1:
		if char in ["M","N","H","A"]:
			if old_style:
				draw.text((currX, textY1-80), char, font=font1, fill=fill)
				currX += font1.getsize(char)[0]
			else:
				draw.text((currX, textY1), char, font=font, fill=fill)
				currX += font.getsize(char)[0]
		else:
			draw.text((currX, textY1), char, font=font, fill=fill)
			currX += font.getsize(char)[0]

	currX = textX2
	for char in line_2:
		if char in ["M", "N", "H"]:
			if old_style:
				draw.text((currX, textY2 - 50), char, font=font1, fill=fill)
				currX += font1.getsize(char)[0]
			else:
				draw.text((currX, textY2), char, font=font, fill=fill)
				currX += font.getsize(char)[0]
		else:
			draw.text((currX, textY2), char, font=font, fill=fill)
			currX += font.getsize(char)[0]

	return im, (textsize1, textsize2)


def generate_2lines_images_m(template, bg='background/bg2.jpg', margin=10, size=(570, 420)):
	#font1 = ImageFont.truetype("Oswald-Regular.ttf", 170)
	font = ImageFont.truetype("MyFont-Regular_ver3.otf", 180)
	if type(bg).__name__ == 'str':
		im = Image.open(bg)
	else:
		im = bg

	im = im.resize(size)
	width, height = im.size
	draw = ImageDraw.Draw(im)
	line_1, line_2 = template.split('/')

	textsize1 = font.getsize(line_1)
	textX1 = int((width - textsize1[0]) / 2)
	textY1 = int((height / 2 - textsize1[1]) / 2) + margin
	textsize2 = font.getsize(line_2)
	textX2 = int((width - textsize2[0]) / 2)
	textY2 = int(height / 2 + (height / 2 - textsize2[1]) / 2) - margin / 2
	fill = tuple(np.random.randint(20, size=3))

	shadow = tuple(np.random.randint(200, 255, size=3))
	direction = tuple(np.random.randint(-3, 3, size=2))
	old_style = True

	'''currX = textX1
	for char in line_1:
		if char in ["N", "H"]:
			if old_style:
				draw.text((currX, textY1 - 60), char, font=font1, fill=fill)
				currX += font1.getsize(char)[0]
			else:
				draw.text((currX, textY1), char, font=font, fill=fill)
				currX += font.getsize(char)[0]
		else:
			draw.text((currX, textY1), char, font=font, fill=fill)
			currX += font.getsize(char)[0]

	currX = textX2
	for char in line_2:
		if char in ["M", "N", "H"]:
			if old_style:
				draw.text((currX, textY2 - 80), char, font=font1, fill=fill)
				currX += font1.getsize(char)[0]
			else:
				draw.text((currX, textY2), char, font=font, fill=fill)
				currX += font.getsize(char)[0]
		else:
			draw.text((currX, textY2), char, font=font, fill=fill)
			currX += font.getsize(char)[0]

	'''
	# Add drop shadow line 1
	#draw.text((textX1 + direction[0], textY1 + direction[1]), line_1, font=font, fill=shadow)
	draw.text((textX1, textY1), line_1, font=font, fill=fill)

	# Add drop shadow line 2
	#draw.text((textX2 + direction[0], textY2 + direction[1]), line_2, font=font, fill=shadow)
	draw.text((textX2, textY2), line_2, font=font, fill=fill)


	return im, (textsize1, textsize2)