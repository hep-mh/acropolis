#! /usr/bin/env bash

# Extract the current working directory
dir=$(basename $PWD)

# Check if the current working directory is correct
if [ "$dir" != "tools" ];
then
    echo "ERROR: This script needs to be executed in the tools/ directory. Stop!"
    exit 1
fi

# Define the data path
data="tools/data"

# Define a function to replace NE_pd and NT_pd
function replace {
    cp $data/params.py.rpl acropolis/params.py
    sed -i "s/__NE_PD__/${1}/" acropolis/params.py
    sed -i "s/__NT_PD__/${2}/" acropolis/params.py
}

# Change the directory
cd ..

# Back up the old param-file
cp acropolis/params.py $data/params.py~

# Scan the different values for NE_pd
for NE_PD in $(cat $data/NE_pd.list); do
    echo $NE_PD

    replace $NE_PD 50
    echo $NE_PD $(./$@) >> $data/NE_pd.dat
done

# Scan the different values for NT_pd
for NT_PD in $(cat $data/NT_pd.list); do
    echo $NT_PD

    replace 150 $NT_PD
    echo $NT_PD $(./$@) >> $data/NT_pd.dat
done

# Restore the old param-file
mv $data/params.py~ acropolis/params.py

# Go back to the original directory
cd tools/
