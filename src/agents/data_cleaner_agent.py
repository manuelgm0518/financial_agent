from agno.agent import Agent
from agno.tools.file import FileTools

from tools import DataCleaningTools

# TODO: Check for día_del_mes_para_hacer_pago column
# TODO: Get information from sheet name

prompt = """
    Please help me process and clean my financial data files. Here's what I need you to do:

    1. FIRST STEP: List all Excel (.xlsx) files in the 'data/' directory.

    2. SECOND STEP: For each file, run an analysis of the data using the analyze_file function.

    3. THIRD STEP: For each file, clean the data using the clean_file function.
        - Set the output_path to 'data_cleaned/[filename]_cleaned.csv'

        - Based on the previous analysis, apply the appropriate cleaning functions and parameters:
            * standardize_columns_kwargs: Parameters for _standardize_columns
                - to_lower (bool): Convert column names to lowercase (default: True)
                - replace_spaces (bool): Replace spaces with underscores (default: True)
                - replace_dashes (bool): Replace dashes with underscores (default: True)
            
            * handle_missing_kwargs: Parameters for _handle_missing_values
                - strategy (str): "drop", "fill", or "none" (default: "drop")
                - fill_value (object): Value to fill if strategy is "fill" (default: None)
                - columns (list): Specific columns to handle (default: None for all)
            
            * remove_duplicates_kwargs: Parameters for _remove_duplicates
                - subset (list): Columns to consider for duplicates (default: None for all)
                - keep (str): "first", "last", or False (default: "first")
            
            * normalize_text_kwargs: Parameters for _normalize_text_columns
                - columns (list): Specific text columns to normalize (default: None for all)
                - to_lower (bool): Convert to lowercase (default: True)
                - strip (bool): Strip whitespace (default: True)
                - normalize_spaces (bool): Normalize multiple spaces (default: True)
            
            * normalize_numeric_kwargs: Parameters for _normalize_numeric_columns
                - columns (list): Specific numeric columns to normalize (default: None for all)
                - decimals (int): Number of decimal places to round to (default: 2)
            
            * convert_dates_kwargs: Parameters for _convert_dates
                - columns (list): Specific date columns to convert (default: None for inferred)
                - format (str): Date format string like "%Y-%m-%d" (default: None for auto-detect)
                - errors (str): "raise", "coerce", or "ignore" (default: "coerce")
            
            * convert_categories_kwargs: Parameters for _convert_categories
                - columns (list): Specific columns to convert to categorical (default: None for inferred)

    4. FOURTH STEP: Provide a summary of what was processed and cleaned.

    Important notes:
    - Always complete all the steps.
    - Make sure to check if files exist before processing.
    - Create the 'data_cleaned/' directory if it doesn't exist.
    - Handle any errors gracefully and continue with other files if one fails
    - Provide clear feedback on what was processed successfully and what failed.
    - Use the analysis results to make informed decisions about cleaning parameters:
        * For missing values: Choose "drop" if few missing values, "fill" if many missing values
        * For dates: Use appropriate format based on the data analysis
        * For numeric columns: Choose appropriate decimal places based on the data
        * For text columns: Consider the data characteristics when choosing normalization options
    - Document your decisions and reasoning for the cleaning parameters chosen.

    Please start with step 1 and proceed step by step.
    """

data_cleaner_agent = Agent(
    name="Data Cleaner Agent",
    description="Clean financial data files",
    tools=[
        FileTools(),
        DataCleaningTools()
        ], 
    show_tool_calls=True,
    instructions=prompt,
)