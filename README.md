# photochop

a tool to chop pngs out of images

## installation
set up a virtual environment with `virtualenv venv' and activate it with `. venv/bin/activate`

then just install all the dependencies `pip install -r requirements.txt`


## usage
`python photochop.py --pre-smooth --minimum-group-size 25 /path/to/image`



```
usage: photochop.py [-h] [--max-subprocesses MAX_SUBPROCESSES]
                    [--disable-diacritics] [--auto-align]
                    [--set-threshold-to SET_THRESHOLD_TO]
                    [--allow-diagonal-connections] [--disable-multiprocessing]
                    [--row-despeckle-size ROW_DESPECKLE_SIZE]
                    [--minimum-group-size MINIMUM_GROUP_SIZE] [--despeckle]
                    [--supercontrast]
                    [--min-groups-per-row MIN_GROUPS_PER_ROW] [--pre-smooth]
                    [--smoothing-passes SMOOTHING_PASSES]
                    filename

dice an image, separating out groups of dark pixels

positional arguments:
  filename              an input png file

optional arguments:
  -h, --help            show this help message and exit
  --max-subprocesses MAX_SUBPROCESSES
                        number of worker processes to use. uses the cpu core
                        count by default
  --disable-diacritics  disable diacritic alignment
  --auto-align          auto-align the document
  --set-threshold-to SET_THRESHOLD_TO
                        set the threshold for a match (0-255)
  --allow-diagonal-connections
                        allow diagonal connections as well as cardinal
  --disable-multiprocessing
                        disable multiprocessing.
  --row-despeckle-size ROW_DESPECKLE_SIZE
                        despeckler size for row chopping
  --minimum-group-size MINIMUM_GROUP_SIZE
                        minimum pixel group size
  --despeckle           despeckle the document.
  --supercontrast       supercontrast the image. can help mitigate compression
                        artifacts.
  --min-groups-per-row MIN_GROUPS_PER_ROW
                        minimum groups per row
  --pre-smooth          smooth original document before processing
  --smoothing-passes SMOOTHING_PASSES
                        number of smoothing passes to make
```
