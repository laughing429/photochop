#! /usr/bin/env python

# a tool that chops a photo into like-pixel groups
# @author: patrick kage

# this program makes some assumptions about the input file
# 1. the text is arranged in rows
# 2. the text is in a single column layout

from scipy import misc, stats, ndimage
import numpy as np
import os, uuid, time, sys, math
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy


class Photochopper:
	def __init__(self, orig_image, threshold):
		# load the image in from a file
		self.original = misc.imread(orig_image, flatten=True);
		print("original shape = " + str(self.original.shape));

		# create an array of zeroes to store which pixels we've already hit
		self.hits = np.zeros(self.original.shape);

		# create an array that'll hold the groups we've already identified
		self.groups = [];

		# set the threshold
		self.threshold = threshold;

		# process diacritics
		self.diacritics_enabled = True;

		# auto-align document
		self.auto_align = False;

		# allow diagonals
		self.diagonal_connections = False;

		# multiprocessing enabled
		self.multiprocessing_enabled = True;

		# set row despeckle size
		self.row_despeckle_size = 3;

		# set minimum group size
		self.minimum_group_size = 5;

		# set the threadcount
		try:
			self.threadcount = cpu_count();
		except NotImplementedError:
			self.threadcount = 8;
			print('cpu_count() is not implemented on this system.');

	def set_max_threads(self, threads):
		self.threadcount = threads;

	def set_threshold(self, val):
		self.threshold = val;

	def enable_diagonals(self, val):
		self.diacritics_enabled = val;

	def enable_diacritics(self, val):
		self.diacritics_enabled = val;

	def enable_auto_align(self, val):
		self.auto_align = val;

	def enable_multiprocessing(self, val):
		self.multiprocessing_enabled = val;

	def set_row_despeckle_size(self, val):
		self.row_despeckle_size = val;

	def set_minimum_group_size(self, val):
		self.minimum_group_size = val;

	def process(self):
		print("supercontrasting...");
		self.__supercontrast();
		print("despeckling...");
		self.__fast_despeckle();
		print("done.");

		if self.auto_align:
			self.__auto_align_document();

		# store all the vertical ranges that have text on them
		rows = [];


		print('extracting characters...');

		print('\tgrabbing rows...');

		if not self.multiprocessing_enabled:
			current = [0,0];
			active = False;
			# do a first pass along the image to find ranges of row with stuff on them
			for y in range(0, self.original.shape[0]):

				thisrow = False;
				for x in range(0, self.original.shape[1]):
					if self.original[y][x] < self.threshold:
						thisrow = True;
						break;
				#print('row ' + str(y) + ' is ' + ('empty' if not thisrow else 'not empty'));
				if thisrow:
					if not active:
						active = True;
						current[0] = y;
				else:
					if active:
						active = False;
						current[1] = y;
						sys.stdout.write('\r\t\tfound a range from Y:%6d to Y:%6d' % (current[0], current[1]));
						sys.stdout.flush();
						rows.append(deepcopy(current));


			if active:
				current[1] = y;
				sys.stdout.write('\r\t\tfound a range from Y:%6d to Y:%6d' % (current[0], current[1]));
				sys.stdout.flush();
				rows.append(deepcopy(current));
		else:
			# create the thread pool
			threads = Pool(self.threadcount);
			print("\t\tcreated pool of up to " + str(self.threadcount) + " threads\n\t\tchunking rows...");
			row_finder = partial(process_rows, self.threshold, self.row_despeckle_size);
			row_chunks = [];
			for i in range(0, self.threadcount):
				frac = self.original.shape[0] / self.threadcount;
				row_chunks.append((self.original[i * frac:(i + 1) * frac], i * frac));

			print("\t\tgenerating row flags...");
			row_flags = threads.map(row_finder, row_chunks);
			print("\t\tconsolidating row flags...");
			allflags = [];
			for row_flag in row_flags:
				allflags += row_flag;

			print('\t\tgenerating flags...');
			current = [0,0];
			active = False;

			for i in range(0, len(allflags)):
				flag = allflags[i];
				if flag and not active:
					active = True;
					current[0] = i;
				elif not flag and active:
					active = False;
					current[1] = i;
					rows.append(deepcopy(current));




		# commence shoehorned multithreading
		print('\n\t\t%d rows found.\n\tslicing data for multithreading...' % len(rows));
		slices = [];
		for row in rows:
			slices.append((self.original[row[0]:row[1]], row[0]));


		if self.multiprocessing_enabled:


			# create a partial
			processor = partial(process_row, self.threshold, self.diacritics_enabled, self.diagonal_connections, self.minimum_group_size);
			print("\tprocessing....");
			tmpgroups = threads.map(processor, slices);
		else:
			tmpgroups = [];
			print('\tstarting sequential processing...');
			for slice in slices:
				tmpgroups.append(process_row(self.threshold, self.diacritics_enabled, self.diacritics_enabled, self.minimum_group_size, slice));

		for x in tmpgroups:
			self.groups += x;
		print('\n\tdone processing.');

	def export_groups(self, dir_name):
		if not os.path.exists('out'):
			os.mkdir('out');


		# make the final output square
		def make_square(grp):
			# figure out the final size of the matrix and fill it with zeroes
			final_size = max(grp.shape[0], grp.shape[1]);
			final = np.zeros((final_size, final_size));
			final.fill(255);

			# starting position (where to start writing the matrix)
			starting = [0,0];
			if final_size == grp.shape[0]:
				starting[1] = (final_size/2) - (grp.shape[1]/2);
			else:
				starting[0] = (final_size/2) - (grp.shape[0]/2);


			# write the matrix
			for y in range(0, grp.shape[0]):
				for x in range(0, grp.shape[1]):
					final[y + starting[0]][x + starting[1]] = grp[y][x];

			return final;


		os.mkdir('out/' + dir_name);
		print('created directory out/' + dir_name);

		i = 0;
		for group in self.groups:
			misc.imsave('out/' + dir_name + '/' + str(i) + '.png', make_square(group));
			i += 1;
		print('saved all groups');

	def __auto_align_document(self):
		print('aligning document...');
		points = [];
		for x in range(0, self.original.shape[1]):
			for y in range(0, self.original.shape[0]):
				if self.original[y][x] < self.threshold:
					points.append((y,x));
					break;
			sys.stdout.write("\r\tgathering data points for doc alignment: %06f%%" % (float(x * 100)/ float(self.original.shape[1])));
			sys.stdout.flush();
		print('\n\tfixing up data...');
		# y's multiplied by -1 because row/col -> x,y
		yvalues = [i[0] * -1 for i in points];

		# clean up the data first
		mean = np.mean(yvalues);
		stddev = np.std(yvalues);

		throw_count = 0;
		to_delete = [];
		for i in range(0, len(points)):
			if abs(yvalues[i] - mean) > stddev/8 and yvalues[i] - mean >= 0:
				to_delete.append(i);

				throw_count += 1;

		for tdel in reversed(to_delete):
			del points[tdel];

		print('\tthrew out %d points outside .125 standard deviations (lb: %d, la: %d, stddev: %f, mean: %f)\n\tfinding linear regression...' % (throw_count, len(yvalues), len(points), stddev, mean));
		yvalues = [i[0] * -1 for i in points];
		xvalues = [i[1] for i in points];



		slope, intercept, r_value, p_value, std_err = stats.linregress(xvalues, y=yvalues);

		angle = math.asin(self.original.shape[0] * float(slope)/self.original.shape[0]);
		print('\t\tangle: ' + str(angle) + ' (rvalue: ' + str(r_value) + ', slope: ' + str(slope) + ')');

		print('\trotating...');
		self.original = ndimage.rotate(self.original, angle, cval=255);

		print('\t\tnew shape: ' + str(self.original.shape));

	def __fast_despeckle(self):
		self.original = ndimage.binary_closing(self.original);

		self.original = np.multiply(self.original, 255);

		self.original = self.original[1:];
		self.original = self.original[:-1];

		for y in range(0, self.original.shape[0]):
			self.original[y][0] = 255;
			self.original[y][self.original.shape[1] - 1] = 255;

		misc.imsave("test.png", self.original);

	def __supercontrast(self):
		ocopy = np.zeros(self.original.shape);
		for y in range(0, self.original.shape[0]):
			for x in range(0, self.original.shape[1]):
				ocopy[y][x] = 255 if self.original[y][x] > self.threshold else 0;

		self.original = ocopy;

		misc.imsave("test_supercontrast.png", ocopy);



# this is because of multithreading
def process_row(threshold, enable_diacritics, enable_diagonals, minimum_group_size, original):
	groups = [];
	final = [];
	y_offset = original[1];
	original = original[0];
	hits = np.zeros(original.shape);

	# look for connected pixels
	for x in range(0, original.shape[1]):
		for y in range(0, original.shape[0]):
			if original[y][x] < threshold and hits[y][x] == 0:
				# (re)create a sparse array
				group = _SparseArray();
				# make sure we look around the origin point
#				group.set(y,x,-1);

				open_pixels = [_Pixel(y,x,-1)];
				# this should be better but it isn't
				while True:
					# pixels to look at next
					to_add = [];
					for i in range(0, len(open_pixels)):

						pixel = open_pixels[i];
						# there should be a better way of doing this
						orig = original[y][x];
						group.set(pixel.y, pixel.x, orig);



						# search around the pixel
						for sy in range(-1, 2):
							for sx in range(-1, 2):
								if abs(sx) + abs(sy) == 2 and not enable_diagonals:
									continue;
								try:
									# and group.get(pixel.y + sy, pixel.x + sx) == None
									if original[pixel.y + sy][pixel.x + sx] < threshold and hits[pixel.y + sy][pixel.x + sx] == 0:
										#group.set(pixel.y + sy, pixel.x + sx, -1);
										to_add.append(_Pixel(pixel.y + sy, pixel.x + sx, -1));
										hits[pixel.y + sy][pixel.x + sx] = 1;

								except:
									pass; # shhhhh... it's okay.

					open_pixels = to_add;
					if len(open_pixels) == 0:
						break;

				# set the pixels we've hit into the hit group
				for pxl in group.arr:
					hits[pxl.y][pxl.x] = 1;

				points = group.get_bounding_points();
				points[0][0] += y_offset;
				points[1][0] += y_offset;
				# brag a little bit
				#sys.stdout.write('\r\t\tidentified a group of %6d pixels spanning from %s to %s         ' % (len(group.arr), points[0], points[1]));
				#sys.stdout.flush();
				groups.append(deepcopy(group));

	# remove smaller groups
	passed_groups = [];
	for group in groups:
		if group.size() >= minimum_group_size:
			passed_groups.append(group);
	groups = passed_groups;

	if enable_diacritics:
		# align diacritics
		addedMultipart = False;
		for i in range(0, len(groups) - 1):
			if addedMultipart:
				addedMultipart = False;
				continue;

			# TODO: make this combine diacretics as well
			r1 = groups[i].get_shape();
			r2 = groups[i + 1].get_shape();
			r1 = set(range(r1[1], r1[3]));
			r2 = set(range(r2[1], r2[3]));

			if len(r1.intersection(r2)) != 0:
				group = _SparseArray();
				group.integrate(groups[i]);
				group.integrate(groups[i + 1]);
				final.append(group.export());
				addedMultipart = True;
			else:
				final.append(groups[i].export());
	else:
		for group in groups:
			final.append(group.export());

	sys.stdout.write('\r\t\t(thread finished) for row at ' + str(y_offset));
	sys.stdout.flush();
	return final;


def process_rows(threshold, row_despeckle_size, bundle):
	chunk = bundle[0];
	y_offset = bundle[1];
	out = [];
	for y in range(0, chunk.shape[0]):
		empty = False;
		speck_count = 0;
		for x in range(0, chunk.shape[1]):
			if chunk[y][x] < threshold:
				if speck_count == row_despeckle_size:
					empty = True;
					break;
				else:
					speck_count += 1;
			else:
				speck_count = 0;
		out.append(empty);
	return out;


# Class to hold a pixel value
class _Pixel():
	def __init__(self, y, x, val):
		self.x = x;
		self.y = y;
		self.val = val;

# sparse array
class _SparseArray():
	def __init__(self):
		self.arr = [];

	# gets a specific pixel
	def get(self, y, x):
		for pixel in self.arr:
			if pixel.y == y and pixel.x == x:
				return pixel.val;
		return None;

	# sets a pixel
	def set(self, y, x, val):
		self.arr.append(_Pixel(y, x, val));
#		return None;

	# gets a list of all pixels with a specific value
	def get_all_of(self, val):
		out = [];
		for pixel in self.arr:
			if pixel.val == val:
				out.append(pixel);
		return out;

	# check if the array contains a value
	def contains(self, val):
		out = False;
		for pixel in self.arr:
			if pixel.val == val:
				out = True;
		return out;

	# gets the min y, min x, max y, and max x of the array
	def get_shape(self):
		lowest_x = self.arr[0].x;
		highest_x = self.arr[0].x;
		lowest_y = self.arr[0].y;
		highest_y = self.arr[0].y;

		for pixel in self.arr:
			if pixel.x < lowest_x:
				lowest_x = pixel.x;
			if pixel.x > highest_x:
				highest_x = pixel.x;
			if pixel.y < lowest_y:
				lowest_y = pixel.y;
			if pixel.y > highest_y:
				highest_y = pixel.y;

		return (lowest_y, lowest_x, highest_y, highest_x);

	# converts the shape to a pair of points
	def get_bounding_points(self):
		shape = self.get_shape();
		return [[shape[0], shape[1]], [shape[0], shape[1]]];

	# exports to a numpy array
	def export(self):
		shape = self.get_shape();
		height = shape[2] - shape[0];
		width = shape[3] - shape[1];

		export = np.zeros((height + 1, width + 1));
		export.fill(255);

		for pixel in self.arr:
			export[pixel.y - shape[0]][pixel.x - shape[1]] = pixel.val;

		return export;

	def size(self):
		return len(self.arr);

	# assimilate another sparse array into itself
	def integrate(self, group):
		for pixel in group.arr:
			self.set(pixel.y, pixel.x, pixel.val);

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="dice an image, separating out groups of dark pixels");
	parser.add_argument('filename', type=str, help="an input png file");
	parser.add_argument('--max-subprocesses', type=int, help="number of worker processes to use. uses the cpu core count by default");
	parser.add_argument('--disable-diacritics', action='store_false', required=False, help="disable diacritic alignment");
	parser.add_argument('--auto-align', action='store_true', required=False, help="auto-align the document");
	parser.add_argument('--set-threshold-to', type=int, required=False, help="set the threshold for a match (0-255)");
	parser.add_argument('--allow-diagonal-connections', action='store_true', required=False, help="allow diagonal connections as well as cardinal");
	parser.add_argument('--disable-multiprocessing', action='store_false', required=False, help="disable multiprocessing.");
	parser.add_argument('--row-despeckle-size', type=int, required=False, help="despeckler size for row chopping");
	parser.add_argument('--minimum-group-size', type=int, required=False, help="minimum pixel group size");
	opts = parser.parse_args();

	dicer = Photochopper(opts.filename, 200);
	if opts.max_subprocesses != None:
		dicer.set_max_threads(opts.max_subprocesses);

	if opts.set_threshold_to != None:
		dicer.set_threshold(opts.set_threshold_to);

	if opts.minimum_group_size != None:
		dicer.set_minimum_group_size(opts.minimum_group_size);

	if opts.row_despeckle_size != None:
		dicer.set_row_despeckle_size(opts.row_despeckle_size);

	dicer.enable_diagonals(opts.allow_diagonal_connections);
	dicer.enable_diacritics(opts.disable_diacritics);
	dicer.enable_auto_align(opts.auto_align);
	dicer.enable_multiprocessing(opts.disable_multiprocessing);

	start_time = time.clock();
	dicer.process();
	end_time = time.clock();
	print('processing took ' + str(end_time - start_time) + ' seconds.');

	start_time = time.clock();
	dicer.export_groups(str(uuid.uuid4()));
	end_time = time.clock();
	print('saving took ' + str(end_time - start_time) + ' seconds.');
