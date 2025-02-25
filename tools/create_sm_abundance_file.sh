#! /usr/bin/env bash

if [ ! -d "alteralterbbn" ]; then
    echo -e "\x1B[38;5;69mCloning AlterAlterBBN...\x1B[0m"

    git clone https://github.com/hep-mh/alteralterbbn.git

    echo
fi
# -->
cd alteralterbbn


if [ ! -f "alteralterbbn/bin/alteralterbbn" ]; then
    echo -e "\x1B[38;5;69mBuilding AlterAlterBBN...\x1B[0m"

    ./build.sh

    echo
fi


echo -e "\x1B[38;5;69mRunning AlterAlterBBN...\x1B[0m"

./bin/alteralterbbn