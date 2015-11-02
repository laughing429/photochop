#! /usr/bin/env python

import sys

if len(sys.argv) != 2:
    print('wrong args');
    sys.exit(1)

text = open(sys.argv[1], 'r').read();
text = ''.join(text.split(' '));
open(sys.argv[1], 'w').write(text);
