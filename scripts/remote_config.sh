#!/usr/bin/env bash

# Project location shared by the remote machines.
remote_project_dir="/Vrac/21500050/dAgger-RLPD-STK"

# Replace these entries with your SSH aliases or user@hostname values.
machines=(
	"ppti-14-302-01"
	# "ppti-14-302-02"
	# "ppti-14-302-03"
	# "ppti-14-302-04"
	# "ppti-14-302-05"
	# "ppti-14-302-07"
	# "ppti-14-302-08"
	# "ppti-14-302-09"
	# "ppti-14-302-10"
	# "ppti-14-302-11"
	# "ppti-14-302-12"
	# "ppti-14-302-13"
	# "ppti-14-302-14"
	# "ppti-14-302-15"
	# "ppti-14-302-16"
)

# Seeds correspond to machines in the same order.
seeds=(
	0
	1
	2
	3
	4
	5
	6
	7
	8
	9
	10
	11
	12
	13
	14
)
