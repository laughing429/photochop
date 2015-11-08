#! /usr/bin/env python

import os, json, sys
from unidecode import unidecode

idir = sys.argv[1];

out = {}
for case in os.listdir(idir):
	if case == '.DS_Store':
		continue;
	out[case] = {};
	for tag in os.listdir(os.path.join(idir, case)):
		if tag == '.DS_Store':
			continue;
		out[case][unidecode(tag)] = [];
		for f in os.listdir(os.path.join(idir, case, tag)):
			if f == '.DS_Store':
				continue;
			out[case][unidecode(tag)].append(f);
			print(f + '\t' + case + '\t' + unidecode(tag[-1]))

json.dump(out, open(idir + '.json', 'w'));
