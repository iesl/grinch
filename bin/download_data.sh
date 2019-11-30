#!/usr/bin/env bash

set -exu

$GRINCH_ROOT/bin/data_processing/download_aloi.sh
$GRINCH_ROOT/bin/data_processing/download_glass.sh