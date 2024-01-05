import pandas as pd
import vobject

# pip install vobject pandas openpyxl

# Read the Excel file into a DataFrame
file_path = 'COSMET-23 Chess Competition (Responses) - Copy.xlsx'
df = pd.read_excel(file_path)

# Create a vCard for each entry and save to a VCF file
vcf_file_path = 'Chess_ppls.vcf'
with open(vcf_file_path, 'w', encoding='utf-8') as vcf_file:
    for index, row in df.iterrows():
        full_name = f"{row['Full Name']} (MSC, CHESS)"
        email = row['Email Address']
        mobile_number = str(row['Contact Information'])  # Convert to string

        # Create a vCard object
        vcard = vobject.vCard()
        vcard.add('FN').value = full_name
        vcard.add('EMAIL').value = email
        vcard.add('TEL').value = mobile_number  # TEL is used for phone numbers

        # Write the vCard to the file
        vcf_file.write(vcard.serialize())

print(f"Contact cards saved to {vcf_file_path}")
