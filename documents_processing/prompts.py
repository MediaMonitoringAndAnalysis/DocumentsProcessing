image_description_prompt = """
I'm going to provide you an image from a humanitarian report. Your objective is to create a text that covers all the analytical information present in the image.
More specificaly, all numbers, specific populations, locations, and any other relevant information should be included in the text.
Make the text as detailed as possible. Use a descriptive language and only mention the information that is present in the image as informative text and information.
The text should be in the form of self-contained paragraphs, and should not be a list of bullet points. Do not provide any general description of the image or introductory text. Instead, directly present the information.
When providing information, present it without reffering the image. For example, instead of saying "The age and gender breakdown of the population indicates that 1% are aged 60 years or above", say "1% are aged 60 years or above".
If the image does not contain any relevant information for the humanitarian report (faces, landscapes, logos, etc.), return an empty string ('-').
Return all text in the language of the image, or in English if the language is not found.
"""

table_description_prompt = """
I'm going to provide you with atable. Your objective is to create a text that covers all the information present in the table.
More specificaly, all numbers, specific populations, locations, and any other relevant information should be included in the text.
Make the text as detailed as possible. Use a descriptive language and only mention the information that is present in the table as informative text and information.
The text should be in the form of self-contained paragraphs, and should not be a list of bullet points. Do not provide any general description of the table or introductory text. Instead, directly present the information.
When providing information, present it without reffering the table. For example, instead of saying "The age and gender breakdown of the population indicates that 1% are aged 60 years or above", say "1% are aged 60 years or above".
Return all text in the language of the table, or in English if the language is not found.
"""

metadata_extraction_prompt = """This is a page of a document. I want to extract the document metadata from this page.
Extract the document publishing date, author organisations and the document title. 
Return only the results in a dictionnary JSON response without unnecessary spaces in the following format:
{
    "date": dd/mm/yyyy,
    "author": List[str]: The author organisations,
    "title": str: The title of the document,
    "type": str: The type of the document: Literal = ["Analysis", "Appeal", "Assessment", "Evaluation and Lessons Learned", "Infographic", "Manual and Guideline", "Map", "News and Press Release", "Other", "Situation Report", "UN Document"]
    "primary_country": str: The primary country of the document,
} 
If you cannot find any of the information, leave the field empty ('-').
Extract the information yourself and do not rely on any external library."""

interview_metadata_extraction_prompt = """This is a page of an interview document. I want to extract the document metadata from this page.
Extract the document publishing date, author organisations and the document title. 
Return only the results in a dictionnary JSON response without unnecessary spaces in the following format:
{
    "date": dd/mm/yyyy: the data when the interview is performed,
    "author": List[str]: The author organisations of the document,
    "title": str: The title of the document,
    "interviewee": List[Dict[str, str]]: The people explicitly mentioned in the document with keys:
        - "name": str: The name of the interviewee,
        - "organization": str: The organization of the interviewee,
        - "role": str: The role of the interviewee in the organization,
        - "organisation_type": str: The type of organization (`Government` / `International Organisation` / `Non Governmental Organisation` / `Media` / `Operating Partner` / `UN agency` / `Donor` / `External` / `Other`),
        - "gender": str: The gender of the interviewee (`M` or `F`, or `M-F` if gender neutral pronoun),
        - "location": str: The location of the interviewee (country level),
        - "location_type": str: The type of location (`global level` / `field-level` / `country-level`),
If you cannot find any of the information, leave the field empty ('-').
Extract the information yourself and do not rely on any external library. 
If you are not sure about an answer, return it anyways. High recall is more important than high precision."""


system_prompts = {
    "Picture": image_description_prompt,
    "Table": table_description_prompt,
}
