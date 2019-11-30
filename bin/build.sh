#!/usr/bin/env bash

set -exu

pushd $GRINCH_ROOT
mvn clean package
popd