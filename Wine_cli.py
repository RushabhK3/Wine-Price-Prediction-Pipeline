



import argparse
from Pipeline import process

parser = argparse.ArgumentParser()

parser.add_argument('input_csv', help='Insert the filepath of the input csv')

# Read arguments on the command line
args = parser.parse_args()
# print(f'Your chosen parameter is {args.Parameter}')

# Call the function process with the csv file as input
new_data = process(args.input_csv)













