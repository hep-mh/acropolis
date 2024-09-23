#! /usr/bin/env bash

# Define a function that is called when 'Ctrl+C' presses
function control_c {
    # Cleanup
    if [ "$dir" == "tools" ] && [ -f "$dir/data/params.py~" ]; then
        # Go back to the original directory
        cd $dir

        # Restore the original 'params.py' file
        mv data/params.py~ ../acropolis/params.py

        # Remove unfinished data files
        rm -f data/NE_pd.dat data/NT_pd.dat
    fi

    # Exit
    exit
}
# -->
trap control_c SIGINT


# Define a function to replace NE_pd and NT_pd
# in the original 'params.py' file
function replace {
    cp $data/params.py~ acropolis/params.py

    sed -i "s/^NE_pd.*$/NE_pd = ${1}/" acropolis/params.py
    sed -i "s/^NT_pd.*$/NT_pd = ${2}/" acropolis/params.py
}


# Define a function to extract the deuterium
# abundances from the output of ACROPOLIS
function extract_deuterium {
    echo "${1}" | awk '/H2/ {print $4 " " $6 " " $8}'
}


# START #######################################################################

cmd_flag=1
# Check if there are a least 8 command-line arguments,
# the first of which is either 'decay' or 'annihilation'
if [ $# -ge 7 ]; then
    if [ "$1" == "decay" ] || [ "$1" == "annihilation" ]; then
        cmd_flag=0
    fi
fi
# -->
# Stop if the previous check did not succeed
if [ $cmd_flag == 1 ]; then
    echo "ERROR: The command-line arguments must be either 'decay [...]' or 'annihilation [...]'. Stop!"
    exit 1
fi

# Extract the current working directory
dir=$(basename $PWD)

# Check if the current working directory is 'acropolis/tools'
if [ "$dir" != "tools" ]; then
    echo "ERROR: This script needs to be executed in the tools/ directory. Stop!"
    exit 1
fi

# Define the data directory
data="$dir/data"

# Cleanup files from previous runs
rm -f $data/NE_pd.dat $data/NT_pd.dat

# Change the directory
cd ..

# Back up the original 'params.py' file
cp acropolis/params.py $data/params.py~

# Scan over the different values for NE_pd
echo "NE_pd"
for NE_PD in $(cat $data/NE_pd.list); do
    # Adjust NE_pd
    replace $NE_PD 50

    # Run ACROPOLIS...
    result=$(./$@)
    # ...and extract the deuterium abundance
    Y2H=$(extract_deuterium "$result")

    echo $NE_PD $Y2H 2>&1 | tee -a $data/NE_pd.dat
done

# Scan over the different values for NT_pd
echo "NT_pd"
for NT_PD in $(cat $data/NT_pd.list); do
    # Adjust NT_pd
    replace 150 $NT_PD

    # Run ACROPOLIS...
    result=$(./$@)
    # ...and extract the deuterium abundance
    Y2H=$(extract_deuterium "$result")

    echo $NT_PD $Y2H 2>&1 | tee -a $data/NT_pd.dat
done

# Restore the original 'params.py' file
mv $data/params.py~ acropolis/params.py

# Go back to the original directory
cd tools/

# END #########################################################################
