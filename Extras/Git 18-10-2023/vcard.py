import pandas as pd
import vobject

#pip install vobject pandas numpy openpyxl

# Read the Excel file into a DataFrame
file_path = 'your_excel_file.xlsx'
df = pd.read_excel(file_path)

# Filter the DataFrame based on the condition (Year != '2nd Year')
filtered_df = df[df['Year'] != '2nd Year']

# Create a vCard for each entry and save to a VCF file
vcf_file_path = 'git_ppls.vcf'
with open(vcf_file_path, 'w', encoding='utf-8') as vcf_file:
    for index, row in filtered_df.iterrows():
        full_name = row['Full Name']
        batch = row['Batch']
        contact_card = f"{full_name} (MSC, JNR, {batch})"
        
        # Create a vCard object
        vcard = vobject.vCard()
        vcard.add('FN').value = contact_card

        # Write the vCard to the file
        vcf_file.write(vcard.serialize())

print(f"Contact cards saved to {vcf_file_path}")
