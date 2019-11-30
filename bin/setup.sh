#!/usr/bin/env bash

export GRINCH_ROOT=`pwd`
export GRINCH_JARPATH=$GRINCH_ROOT/target/xcluster-0.1-SNAPSHOT-jar-with-dependencies.jar
export PYTHONPATH=$GRINCH_ROOT/src/python:$PYTHONPATH
export PATH=$GRINCH_ROOT/dep/apache-maven-3.6.1/bin:$PATH