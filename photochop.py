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

		# enable despeckling
		self.despeckle_enabled = False;

		# enable supercontrasting
		self.supercontrasting_enabled = False;

		# min groups per row
		self.min_groups_per_row = 5;

		# enable pre-smoothing
		self.pre_smooth = False;

		# smoothing passes
		self.smoothing = 1;

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

	def enable_supercontrasting(self, val):
		self.supercontrasting_enabled = val;

	def enable_despeckling(self, val):
		self.despeckle_enabled = val;

	def set_min_groups_per_row(self, val):
		self.min_groups_per_row = val;

	def enable_pre_smoothing(self, val):
		self.pre_smooth = val;

	def set_smoothing_passes(self, val):
		self.smoothing = val;

	def process(self):
		
		# despeckle if enabled
		if self.despeckle_enabled:
			print("despeckling...");
			self.__fast_despeckle();

		# autoalign if enabled
		if self.auto_align:
			self.__auto_align_document();

		# pre smooth if enabled
		if self.pre_smooth:
			print("pre-smoothing...");
			self.original = self.__smooth_group(self.original);


		# supercontrast if enabled
		if self.supercontrasting_enabled:
			print("supercontrasting...");
			self.__supercontrast();


		misc.imsave('test_post_pre_processed.png', self.original);		

		print('extracting characters...');

		# invert the image - makes rowfinding work properly
		for y in range(self.original.shape[0]):
			for x in range(self.original.shape[1]):
				self.original[y][x] = not self.original[y][x];

		# label the original image - each group has a unique label
		lbls, nlbls = ndimage.measurements.label(self.original);

		print('\tnumber of characters:' + str(nlbls));
		print('\tcreating groups...');


		# collect the points from each label
		seen = {};
		for y in range(lbls.shape[0]):
			for x in range(lbls.shape[1]):
				curr = lbls[y][x];
				if not curr in seen:
					seen[curr] = [];
				seen[curr].append((y,x));

		# dump an image representation of the tagged file
		# misc.imsave("test2.png", np.multiply(lbls, 255.0/float(nlbls)));


		# convert the points we collected earlier into sparse array objects for 
		# more complex processing 
		print('\tprocessing groups');
		groups = [];
		for key in seen:
			#sys.stdout.write(str(key) + " ");
			groups.append(_SparseArray());
			groups[-1].arr = seen[key];
		print('\tprocessed groups');
		groups = groups[1:];
		
		

		# create a array to store the number of characters in each row
		row_flags = np.zeros([self.original.shape[0]]);

		# determine the number of groups a particular y-value intersects
		for group in groups:
			bpoints = group.get_shape();
			print(bpoints);
			for i in range(bpoints[0], bpoints[2]):
				row_flags[i] += 1;


		# generate row definitions from the flags we collected earlier
		rows = []; curr = False; start = 0;
		for i in range(0, len(row_flags)):
			if not curr and row_flags[i] > self.min_groups_per_row:
				curr = True;
				start = i;
			elif curr and row_flags[i] <= self.min_groups_per_row:
				rows.append((start, i));
				curr = False;
			#sys.stdout.write(str(int(row_flags[i])) + " ");

		# dump the rows we found onto stdout
		# print('\n\nrows:');
		# for row in rows:
		# 	sys.stdout.write(str(row) + " ");


		# what the print says
		print('\ngenerating group lists from regions...');

		# preinitializing the regions
		regions = {};
		for i in range(0, len(rows)):
			regions[i] = [];

		# figuring out which groups fit into which regions
		for grp in groups:
			bpoints = grp.get_shape();
			#sys.stdout.write(str(bpoints) + " ");
			cset = set(range(bpoints[0], bpoints[2]));
			for i in range(0, len(rows)):
				row = rows[i];
				# if the group intersects the row and the group is over the minimum size then append it to
				# the region currently being analysed
				if cset.intersection(set(range(row[0], row[1]))) and grp.size() > self.minimum_group_size:
					regions[i].append(grp);

		# sort groups
		print('sorting groups within regions...');
		for key in regions:
			# sort by lowest x-value
			print('groups in region ' + str(key) + ': ' + str(len(regions[key])))
			regions[key] = sorted(regions[key], key=lambda g: g.get_shape()[1]);

		print('combining diacritics...');

		# combine diacritics
		final = [];
		final_regions = {}
		for key in regions:
			final_regions[key] = [];
			addedMultipart = False;
			for i in range(0, len(regions[key]) - 1):
				if addedMultipart:
					addedMultipart = False;
					continue;

				# black magic
				r1 = regions[key][i].get_shape();
				r2 = regions[key][i + 1].get_shape();
				max_distance = float((r1[3] - r1[1]) + (r2[3] - r2[1]));
				distance = float(max(r1[3], r2[3]) - min(r1[1], r2[1]));
				r1 = set(range(r1[1], r1[3]));
				r2 = set(range(r2[1], r2[3]));
				
				#print('distance: ' + str(distance) + '\tmax_distance: ' + str(max_distance) + '\tscore: ' + str(distance/max_distance));
				#  and (distance/max_distance) < .85
				if len(r1.intersection(r2)) > 0:
					group = _SparseArray();
					group.integrate(regions[key][i]);
					group.integrate(regions[key][i + 1]);
					final_regions[key].append(group)
					final.append(group.export());
					addedMultipart = True;
				else:
					final_regions[key].append(regions[key][i]);
					final.append(regions[key][i].export());
			if not addedMultipart:
				final.append(regions[key][-1].export());
				final_regions[key].append(regions[key][-1]);


		# do a quick spacing pass
		import csv

		with open('spacingflags.csv', 'w') as f:
			writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL);
			for key in final_regions:
				out = [];
				for i in range(0, len(final_regions[key]) - 1):
					r1 = final_regions[key][i].get_shape();
					r2 = final_regions[key][i + 1].get_shape();
					out.append(r2[1] - r1[3]);
				writer.writerow(out);


		print('done combining.\nexporting...');
		self.groups = final;
		self.final_regions = final_regions;
		print('primed for export');

		# dumping data into a csv
		

		with open('rowflags.csv', 'w') as f:
			writer = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL);
			for row in row_flags:
				writer.writerow([int(row)]);

		return;


	def process_words(self):
		print('processing word groups...');
		final = {};
		for key in self.final_regions:
			print('\tcurrently processing line ' + str(key) + '...\n\t\tdoing stats analysis pass...');
			spacing = [];
			for i in range(0, len(self.final_regions[key]) - 1):
				r1 = self.final_regions[key][i].get_shape();
				r2 = self.final_regions[key][i + 1].get_shape();
				spacing.append(r2[1] - r1[3]);

			q3, q1 = np.percentile(spacing, [75 ,25]);
			print("\t\t\tq1: %f, q3: %f" % (q1,q3));
			threshold = 3 * (q3 - q1)
			#sys.stdout.write("outliers: ");
			final[key] = [];
			current = [];
			for i in range(0, len(self.final_regions[key]) - 1):
				r1 = self.final_regions[key][i].get_shape();
				r2 = self.final_regions[key][i + 1].get_shape();

				current.append(self.__make_square(self.final_regions[key][i].export()));
				if r2[1] - r1[3] > q3 + threshold:
					final[key].append(deepcopy(current));
					current = [];
			current.append(self.__make_square(self.final_regions[key][-1].export()));
			final[key].append(deepcopy(current));

		self.words = final;
		
	def dump_words(self, odir):
		i = 0;
		os.system('mkdir dump');
		for key in self.final_regions:
			for grp in self.final_regions[key]:
				misc.imsave('dump/%06d.png' % i, grp.export());
				i += 1;
		
		i = 0;
		crow = 0;
		for key in self.words:
			wordc = 0;
			for word in self.words[key]:
				os.system('mkdir -p %s/row_%d/word_%d' % (odir, crow, wordc));
				for grp in word:
					misc.imsave('%s/row_%d/word_%d/%06d.png' % (odir, crow, wordc, i), self.__make_square(grp));
					i += 1
				wordc += 1
			crow += 1





	def __make_square(self, grp):
		# figure out the final size of the matrix and fill it with zeroes
		final_orig = max(grp.shape[0], grp.shape[1]);
		final_size = float(final_orig if (final_orig % 2) == 0 else final_orig + 1);
		final = np.zeros((final_size, final_size));
		final.fill(255);

		# starting position (where to start writing the matrix)
		starting = [0,0];
		if final_orig == grp.shape[0]:
			starting[1] = (final_orig/2) - (grp.shape[1]/2);
		else:
			starting[0] = (final_orig/2) - (grp.shape[0]/2);

		#print('starting pos: ' + str(starting) + (' (corrected)' if final_orig == final_size else ''));

		# write the matrix
		for y in range(0, grp.shape[0]):
			for x in range(0, grp.shape[1]):
				final[y + starting[0]][x + starting[1]] = grp[y][x];
		return final;

	def export_groups(self):
		# make the final output square
		self.final = [self.__make_square(grp) for grp in self.groups];



	def write_out(self, dir_name):
		if not os.path.exists('out'):
			os.mkdir('out');

		os.mkdir('out/' + dir_name);
		print('created directory out/' + dir_name);

		i = 0;
		for group in self.final:
			#print("working on group " + str(i));
			misc.imsave('out/%s/%06d.png' % (dir_name, i), group);
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
			#if abs(yvalues[i]) > self.original.shape[0]/16:
				to_delete.append(i);
				throw_count += 1;

		for tdel in reversed(to_delete):
			del points[tdel];

		import csv;
		with open('alignpoints.csv', 'w') as f:
			writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL);
			for pt in points:
				writer.writerow(pt);

		print('\tthrew out %d points outside .125 standard deviations (lb: %d, la: %d, stddev: %f, mean: %f)\n\tfinding linear regression...' % (throw_count, len(yvalues), len(points), stddev, mean));
		yvalues = [i[0] * -1 for i in points];
		xvalues = [i[1] for i in points];



		slope, intercept, r_value, p_value, std_err = stats.linregress(xvalues, y=yvalues);

		angle = math.asin(self.original.shape[0] * float(slope)/self.original.shape[0]);
		print('\t\tangle: ' + str(angle) + ' (rvalue: ' + str(r_value) + ', slope: ' + str(slope) + ')');

		print('\trotating...');
		self.original = ndimage.rotate(self.original, angle, cval=255);

		misc.imsave('test_post_rotation.png', self.original);

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

	def __highlight_rows(self, rows):
		print("number of rows to be highlighted: " + str(len(rows)));
		print("rows: " + str(rows));

		self.original = np.multiply(self.original, 255);

		for row in rows:
			print("\tprocessing row " + str(row));
			for y in range(row[0], row[1]):
				for x in range(0, self.original.shape[1]):
					self.original[y][x] = (self.threshold + 5) if self.original[y][x] > self.threshold else self.original[y][x];

		misc.imsave("test_rows.png", self.original);

	def __highlight_bpoints(self, groups):

		for i in range(0, len(groups)):
			group = groups[i];
			#sys.stdout.write(str(i) + "(" + str(group.size()) + ")[" + str(group.get_bounding_points()) + "\t");
			#sys.stdout.flush();
			shape = group.get_shape();
			for x in range(shape[1], shape[3]):
				self.original[shape[0]][x] = 0;
				self.original[shape[2]][x] = 0;
			for y in range(shape[0], shape[2]):
				self.original[y][shape[1]] = 0;
				self.original[y][shape[3]] = 0;

		misc.imsave("test_groups.png", self.original);

	def __smooth_group(self, arr):
		for i in range(0, self.smoothing):
			arr = misc.imfilter(arr, 'smooth');
			
			#arr = ndimage.filters.gaussian_filter(deepcopy(arr), 5, cval=255);
			for y in range(0, arr.shape[0]):
				for x in range(0, arr.shape[1]):
					arr[y][x] = 255 if arr[y][x] > self.threshold else 0;

		return arr;


# sparse array
class _SparseArray():
	def __init__(self):
		self.arr = [];

	# gets a specific pixel
	def get(self, y, x):
		for pixel in self.arr:
			if pixel[0] == y and pixel[1] == x:
				return True;
		return False;

	# sets a pixel
	def set(self, y, x):
		self.arr.append([y,x]);

	# gets the min y, min x, max y, and max x of the array
	def get_shape(self):
		lowest_x = self.arr[0][1];
		highest_x = self.arr[0][1];
		lowest_y = self.arr[0][0];
		highest_y = self.arr[0][0];

		for pixel in self.arr:
			if pixel[1] < lowest_x:
				lowest_x = pixel[1];
			if pixel[1] > highest_x:
				highest_x = pixel[1];
			if pixel[0] < lowest_y:
				lowest_y = pixel[0];
			if pixel[0] > highest_y:
				highest_y = pixel[0];

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
			export[pixel[0] - shape[0]][pixel[1] - shape[1]] = 0;

		return export;

	def size(self):
		return len(self.arr);

	# assimilate another sparse array into itself
	def integrate(self, group):
		for pixel in group.arr:
			self.set(pixel[0], pixel[1]);

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
	parser.add_argument('--despeckle', action='store_true', required=False, help="despeckle the document.");
	parser.add_argument('--supercontrast', action='store_true', required=False, help="supercontrast the image. can help mitigate compression artifacts.");
	parser.add_argument('--min-groups-per-row', required=False, help="minimum groups per row");
	parser.add_argument('--pre-smooth', required=False, action='store_true', help="smooth original document before processing");
	parser.add_argument('--smoothing-passes', required=False, type=int, help="number of smoothing passes to make");
	opts = parser.parse_args();

	dicer = Photochopper(opts.filename, 150);
	if opts.max_subprocesses is not None:
		dicer.set_max_threads(opts.max_subprocesses);

	if opts.set_threshold_to is not None:
		dicer.set_threshold(opts.set_threshold_to);

	if opts.minimum_group_size is not None:
		dicer.set_minimum_group_size(opts.minimum_group_size);

	if opts.row_despeckle_size is not None:
		dicer.set_row_despeckle_size(opts.row_despeckle_size);

	if opts.min_groups_per_row is not None:
		dicer.set_min_groups_per_row(opts.min_groups_per_row);

	if opts.smoothing_passes is not None:
		dicer.set_smoothing_passes(opts.smoothing_passes);

	dicer.enable_diagonals(opts.allow_diagonal_connections);
	dicer.enable_diacritics(opts.disable_diacritics);
	dicer.enable_auto_align(opts.auto_align);
	dicer.enable_multiprocessing(opts.disable_multiprocessing);
	dicer.enable_supercontrasting(opts.supercontrast);
	dicer.enable_despeckling(opts.despeckle);
	dicer.enable_pre_smoothing(opts.pre_smooth);

	start_time = time.clock();
	dicer.process();
	end_time = time.clock();
	print('processing took ' + str(end_time - start_time) + ' seconds.');
	dicer.process_words();
	dicer.dump_words('words');

	start_time = time.clock();
	dicer.export_groups();
	odir = opts.filename.split('/')[-1];
	odir = (odir.split('.')[-2])

	dicer.write_out(odir);
	end_time = time.clock();
	print('saving took ' + str(end_time - start_time) + ' seconds.');
