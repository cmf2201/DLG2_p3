def filter_lines_by_actual_dates(input_file, output_file, valid_dates):
    """
    Filters lines that start with one of the given valid dates in the format 'YYYY_M_D'.

    Args:
    - input_file: path to the input text file.
    - output_file: path to the output text file.
    - valid_dates: list of date patterns (e.g., ['2011_9_26', '2011_9_28', '2011_9_29', '2011_9_30']).
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Iterate over each line in the input file
            for line in infile:
                # Strip any leading/trailing whitespace characters (including newlines)
                line = line.strip()

                # Check if the line starts with any of the valid dates
                if any(line.startswith(date) for date in valid_dates):
                    # Write the line to the output file if it matches
                    outfile.write(line + '\n')

        print(f"Filtered lines successfully written to {output_file}.")
    except Exception as e:
        print(f"Error: {e}")

input_file = "DLG2_p3/src/Src/list/eigen_train_list.txt"
output_file = "DLG2_p3/src/Src/list/filter_eigen_train_list.txt"

# Valid date prefixes in the format 'YYYY_M_D' (based on actual dates in the file)
valid_dates = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30']  # Modify this list as needed
filter_lines_by_actual_dates(input_file, output_file, valid_dates)

print(f"Filtered lines written to {output_file}.")
