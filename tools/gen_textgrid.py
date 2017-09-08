#!/usr/bin/python

# Created on 2016-07-10
# Author: Zhang Binbin

import sys

def print_segment(cur, start, end, info):
    print "\t\tintervals [%d]:" % cur
    print "\t\t\txmin = %f" % start
    print "\t\t\txmax = %f" % end
    print "\t\t\ttext = \"%s\"" % info

for line in sys.stdin:
    arr = line.strip().split()
    filename = arr[0]
    end_time = float(arr[1])
    if len(arr) <= 2: break
    assert((len(arr) - 2) % 4 == 0)
    intervals = ((len(arr) - 2) / 4) * 2 - 1;

    print "File type = \"ooTextFile\""
    print "Object class = \"TextGrid\""
    print "xmin = 0"
    print "xmax = %f" % end_time
    print "tiers? <exists>"
    print "size = 1"
    print "item []:"
    print "\t item [1]:"
    print "\t\tclass = \"IntervalTier\""
    print "\t\tname = \"%s\"" % arr[0]
    print "\t\txmin = 0" 
    print "\t\txmax = %f" % end_time
    seg_arr = arr[2:]
    assert(len(seg_arr) % 4 == 0)
    # beginging silence
    if len(seg_arr) > 4 and int(seg_arr[1]) != 0:
        intervals += 1
    if len(seg_arr) > 4 and int(end_time * 100) != int(seg_arr[-2]):
        intervals += 1
    print "\t\tintervals: size = %d " % intervals

    cur = 1
    if len(seg_arr) > 4 and int(seg_arr[1]) != 0:
        print_segment(cur, 0, int(seg_arr[1]) / 100.0, "N")
        cur += 1

    # middle silence 
    for i in range(0, len(seg_arr), 4):
        print_segment(cur, int(seg_arr[i+1]) / 100.0, int(seg_arr[i+2]) / 100.0, "V")
        cur += 1
        if i < len(seg_arr) - 4:
            print_segment(cur, int(seg_arr[i+2]) / 100.0, int(seg_arr[i+5]) / 100.0, "N")
            cur += 1

    # ending silence
    if len(seg_arr) > 4 and int(end_time * 100) != int(seg_arr[-2]):
        print_segment(cur, int(seg_arr[-2]) / 100.0, end_time, "N")

