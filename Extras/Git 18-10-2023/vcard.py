import pandas as pd
import vobject

#pip install vobject


# Read the Excel file into a DataFrame
file_path = 'your_excel_file.xlsx'
df = pd.read_excel(file_path)

# Filter the DataFrame based on the condition (Year != '2nd Year')
filtered_df = df[df['Year'] != '2nd Year']

# Create a vCard for each entry and save to a vCard file
vcard_file_path = 'git_ppls.vcard'
with open(vcard_file_path, 'w') as vcard_file:
    for index, row in filtered_df.iterrows():
        full_name = row['Full Name']
        batch = row['Batch']
        contact_card = f"{full_name} (MSC, JNR, {batch})"
        
        # Create a vCard object
        vcard = vobject.vCard()
        vcard.add('FN').value = contact_card

        # Write the vCard to the file
        vcard_file.write(vcard.serialize())

print(f"Contact cards saved to {vcard_file_path}")
