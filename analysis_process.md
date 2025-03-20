# Steps to data analysis (destined to users)
## Reference sample process
1. Acquire the data
2. Run the files_io.py file as a script or run the prepare_datafile() function on the datafiles (both options create the config files and directories)
3. Edit the configuration files as they are created to provide the additionnal data necessary for the following steps. The zstart and zstop values must be choosen so that the analysed volume contains only sample. It must be filled with tracers. If the analysed volume contains a section that is below or above the sample, tracer concentration will be underestimated.
4. Launch the process_reference_sample.py file as a script if you want to analyse the file one by one.
5. Alternatively, run the process_reference_sample() function in a loop to analyse several files in a row. This step requires you to include the saving of the results in your loop though.

## Capsules sample
1. Acquire the data
2. Run the files_io.py file as a script or run the prepare_datafile() function on the datafiles (both options create the config files and directories)
3. Edit the configuration files as they are created to provide the additionnal data necessary for the following steps.
4. Run the measure_capsules.py file as a script if you want to analyse the files one by one.
5. Alternatively, run the measure_capsules() function in a loop to analyse several files in a row. This step requires you to include the saving of the results in your loop though.
6. Gather the results from the different datafiles in one dataframe and run the capsule_disrtib_kde() and the capsule_scatter() functions on it to produce the distribution / KDE and scatterplots.