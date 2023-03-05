import csv
import os

# Read the CSV file
with open('../../data/adventures/Zikran’s Zephyrean Tome.csv', 'r') as f:
    reader = csv.reader(f)
    with open(os.path.join(os.path.dirname('../../data/adventures/Zikran’s Zephyrean Tome.csv'), "Zikran’s Zephyrean Tome.md"), 'w') as f:
        for row in reader:
            # In each row, take the column named "Content" and 1. Use the first line as the title, 2. Use the rest of the lines as the content
            # Write the title and content to a markdown file in the same directory as the CSV file
            content = row[1]
            split = content.splitlines()
            title = split[0]
            content = "\n".join(split[1:])
            f.write(f"# {title}\n\n{content}\n\n")
