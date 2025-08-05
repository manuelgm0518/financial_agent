"""
Tools for cleaning and transforming data.
"""

from agno.tools import Toolkit
from agno.utils.log import logger
import pandas as pd
from typing import Optional, Union
from pathlib import Path
import os

class DataCleaningTools(Toolkit):
    """
    A comprehensive toolkit for cleaning and transforming data files.
    
    This class provides a suite of tools for data cleaning operations including:
    - File reading (Excel, CSV)
    - Column name standardization
    - Missing value handling
    - Duplicate removal
    - Text normalization
    - Numeric column rounding
    - Date conversion
    - Categorical conversion
    - Data analysis and insights
    
    The toolkit is designed to work with pandas DataFrames and provides both
    individual cleaning methods and a comprehensive cleaning pipeline.
    
    Attributes:
        name (str): The name of the toolkit ("data_cleaning_tools")
        tools (list): List of available tools (clean_file, analyze_file)
    """
    def __init__(self, **kwargs):
        """
        Initialize the DataCleaningTools toolkit.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the parent Toolkit class
        """
        super().__init__(name="data_cleaning_tools", tools=[
            self.clean_file,
            self.analyze_file,
        ], **kwargs)

    def _read_file(self, file_path: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        Reads a file (Excel or CSV) and returns a pandas DataFrame.
        
        Args:
            file_path (str): Path to the input file. Supported formats: .xlsx, .xls, .csv
            sheet_name (Union[str, int], optional): Sheet name or index to read for Excel files. 
                                                   Defaults to 0 (first sheet).
        
        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file format is not supported
        """
        logger.info(f"Reading file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()

        if file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .xlsx, .xls, .csv")
        return df

    def _standardize_columns(self, df: pd.DataFrame, to_lower: bool = True, replace_spaces: bool = True, replace_dashes: bool = True) -> pd.DataFrame:
        """
        Standardizes column names by converting to lowercase, replacing spaces with underscores, 
        and replacing dashes with underscores.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            to_lower (bool, optional): Whether to convert column names to lowercase. Defaults to True.
            replace_spaces (bool, optional): Whether to replace spaces with underscores. Defaults to True.
            replace_dashes (bool, optional): Whether to replace dashes with underscores. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        cleaned_df = df.copy()
        columns = cleaned_df.columns
        if to_lower:
            columns = columns.str.lower()
        if replace_spaces:
            columns = columns.str.replace(' ', '_')
        if replace_dashes:
            columns = columns.str.replace('-', '_')
        cleaned_df.columns = columns
        return cleaned_df

    def _remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None, keep: str = "first") -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            subset (Optional[list], optional): Column names to consider for identifying duplicates. 
                                             If None, all columns are used. Defaults to None.
            keep (str, optional): Which duplicates to keep. Options: 'first', 'last', False. 
                                Defaults to "first".
        
        Returns:
            pd.DataFrame: DataFrame with duplicate rows removed
        """
        return df.drop_duplicates(subset=subset, keep=keep)

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = "drop", fill_value: Optional[object] = None, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Handles missing values in the DataFrame according to the specified strategy.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str, optional): Strategy for handling missing values. Options: 'drop', 'fill', 'none'. 
                                    Defaults to "drop".
            fill_value (Optional[object], optional): Value to use when filling missing values. 
                                                   If None and strategy is 'fill', empty string is used. Defaults to None.
            columns (Optional[list], optional): Specific columns to apply missing value handling to. 
                                              If None, all columns are processed. Defaults to None.
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled according to the strategy
            
        Raises:
            ValueError: If an unknown strategy is provided
        """
        if strategy == "none":
            return df
            
        if columns is not None:
            # Filter to only include columns that exist in the DataFrame
            existing_columns = [col for col in columns if col in df.columns]
            if not existing_columns:
                logger.warning(f"None of the specified columns {columns} exist in the DataFrame. Available columns: {list(df.columns)}")
                return df
            target_df = df[existing_columns]
        else:
            target_df = df

        if strategy == "drop":
            if columns is not None:
                cleaned_df = df.dropna(subset=existing_columns)
            else:
                cleaned_df = df.dropna()
        elif strategy == "fill":
            if fill_value is None:
                fill_value = ""
            cleaned_df = df.copy()
            if columns is not None:
                cleaned_df[existing_columns] = cleaned_df[existing_columns].fillna(fill_value)
            else:
                cleaned_df = cleaned_df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown missing value handling strategy: {strategy}")
        return cleaned_df

    def _normalize_text_columns(self, df: pd.DataFrame, columns: Optional[list] = None, to_lower: bool = True, strip: bool = True, normalize_spaces: bool = True) -> pd.DataFrame:
        """
        Normalizes text columns by converting to lowercase, stripping whitespace, and normalizing spaces.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[list], optional): Specific text columns to normalize. 
                                              If None, all object (string) columns are processed. Defaults to None.
            to_lower (bool, optional): Whether to convert text to lowercase. Defaults to True.
            strip (bool, optional): Whether to strip leading and trailing whitespace. Defaults to True.
            normalize_spaces (bool, optional): Whether to normalize multiple spaces to single spaces. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with normalized text columns
        """
        cleaned_df = df.copy()
        if columns is None:
            text_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
        else:
            # Filter to only include columns that exist in the DataFrame
            text_columns = [col for col in columns if col in cleaned_df.columns]
            if not text_columns:
                logger.warning(f"None of the specified text columns {columns} exist in the DataFrame. Available columns: {list(cleaned_df.columns)}")
                return cleaned_df
        for col in text_columns:
            cleaned_col = cleaned_df[col].astype(str)
            if strip:
                cleaned_col = cleaned_col.str.strip()
            if to_lower:
                cleaned_col = cleaned_col.str.lower()
            if normalize_spaces:
                cleaned_col = cleaned_col.str.replace(r'\s+', ' ', regex=True)
            cleaned_df[col] = cleaned_col
        return cleaned_df

    def _normalize_numeric_columns(self, df: pd.DataFrame, columns: Optional[list] = None, decimals: int = 2) -> pd.DataFrame:
        """
        Rounds numeric columns to a specified number of decimal places.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[list], optional): Specific numeric columns to normalize. 
                                              If None, all numeric columns are processed. Defaults to None.
            decimals (int, optional): Number of decimal places to round to. Defaults to 2.
        
        Returns:
            pd.DataFrame: DataFrame with normalized numeric columns (rounded to specified decimal places)
        """
        cleaned_df = df.copy()
        if columns is None:
            num_columns = cleaned_df.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter to only include columns that exist in the DataFrame
            num_columns = [col for col in columns if col in cleaned_df.columns]
            if not num_columns:
                logger.warning(f"None of the specified numeric columns {columns} exist in the DataFrame. Available columns: {list(cleaned_df.columns)}")
                return cleaned_df
        for col in num_columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').round(decimals)
        return cleaned_df

    def _convert_dates(self, df: pd.DataFrame, columns: Optional[list] = None, format: Optional[str] = None, errors: str = "coerce") -> pd.DataFrame:
        """
        Converts specified columns to datetime format.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[list], optional): Specific columns to convert to datetime. 
                                              If None, columns with 'date' or 'fecha' in their name are inferred. Defaults to None.
            format (Optional[str], optional): Date format string (e.g., '%Y-%m-%d'). 
                                            If None, pandas will attempt to infer the format. Defaults to None.
            errors (str, optional): How to handle parsing errors. Options: 'raise', 'coerce', 'ignore'. Defaults to "coerce".
        
        Returns:
            pd.DataFrame: DataFrame with date columns converted to datetime format
        """
        cleaned_df = df.copy()
        if columns is None:
            # Try to infer date columns by name
            date_like = [col for col in cleaned_df.columns if "date" in col.lower() or "fecha" in col.lower()]
        else:
            # Filter to only include columns that exist in the DataFrame
            date_like = [col for col in columns if col in cleaned_df.columns]
            if not date_like:
                logger.warning(f"None of the specified date columns {columns} exist in the DataFrame. Available columns: {list(cleaned_df.columns)}")
                return cleaned_df
        for col in date_like:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], format=format, errors=errors)
        return cleaned_df

    def _convert_categories(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Converts specified columns to categorical dtype for memory efficiency and better performance.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[list], optional): Specific columns to convert to categorical. 
                                              If None, object columns with few unique values (< 20 or < 10% of rows) are inferred. Defaults to None.
        
        Returns:
            pd.DataFrame: DataFrame with specified columns converted to categorical dtype
        """
        cleaned_df = df.copy()
        if columns is None:
            # Try to infer categorical columns: object columns with few unique values
            candidates = []
            for col in cleaned_df.select_dtypes(include=['object']).columns:
                nunique = cleaned_df[col].nunique(dropna=False)
                if nunique < max(20, 0.1 * len(cleaned_df)):
                    candidates.append(col)
            columns = candidates
        else:
            # Filter to only include columns that exist in the DataFrame
            columns = [col for col in columns if col in cleaned_df.columns]
            if not columns:
                logger.warning(f"None of the specified categorical columns {columns} exist in the DataFrame. Available columns: {list(cleaned_df.columns)}")
                return cleaned_df
        for col in columns:
            cleaned_df[col] = cleaned_df[col].astype('category')
        return cleaned_df

    def clean_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        standardize_columns_kwargs: Optional[dict] = None,
        handle_missing_kwargs: Optional[dict] = None,
        remove_duplicates_kwargs: Optional[dict] = None,
        normalize_text_kwargs: Optional[dict] = None,
        normalize_numeric_kwargs: Optional[dict] = None,
        convert_dates_kwargs: Optional[dict] = None,
        convert_categories_kwargs: Optional[dict] = None,
    ) -> str:
        """
        Cleans a CSV or Excel file and saves the cleaned data as a CSV.
        
        This method applies a comprehensive cleaning pipeline including:
        - Column name standardization
        - Missing value handling
        - Duplicate removal
        - Text normalization
        - Numeric column rounding
        - Date conversion
        - Categorical conversion
        
        Args:
            file_path (str): Path to the input file (.csv, .xlsx, .xls)
            output_path (Optional[str], optional): Path to save the cleaned CSV. 
                                                 If None, appends '_cleaned.csv' to input filename. Defaults to None.
            standardize_columns_kwargs (Optional[dict], optional): Keyword arguments for _standardize_columns method. Defaults to None.
            handle_missing_kwargs (Optional[dict], optional): Keyword arguments for _handle_missing_values method. Defaults to None.
            remove_duplicates_kwargs (Optional[dict], optional): Keyword arguments for _remove_duplicates method. Defaults to None.
            normalize_text_kwargs (Optional[dict], optional): Keyword arguments for _normalize_text_columns method. Defaults to None.
            normalize_numeric_kwargs (Optional[dict], optional): Keyword arguments for _normalize_numeric_columns method. Defaults to None.
            convert_dates_kwargs (Optional[dict], optional): Keyword arguments for _convert_dates method. Defaults to None.
            convert_categories_kwargs (Optional[dict], optional): Keyword arguments for _convert_categories method. Defaults to None.
        
        Returns:
            str: The path to the saved cleaned CSV file
            
        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If the file format is not supported
        """
        logger.info(f"Cleaning file: {file_path}")
        df = self._read_file(file_path)
        
        # Store original column names for mapping
        original_columns = df.columns.tolist()

        # Standardize columns
        if standardize_columns_kwargs is None:
            standardize_columns_kwargs = {}
        df = self._standardize_columns(df, **standardize_columns_kwargs)
        
        # Create mapping from original to standardized column names
        column_mapping = dict(zip(original_columns, df.columns.tolist()))
        
        # Helper function to map column names
        def map_columns(columns_list):
            if columns_list is None:
                return None
            return [column_mapping.get(col, col) for col in columns_list]

        # Handle missing values
        if handle_missing_kwargs is None:
            handle_missing_kwargs = {}
        else:
            # Map column names if specified
            if 'columns' in handle_missing_kwargs:
                handle_missing_kwargs['columns'] = map_columns(handle_missing_kwargs['columns'])
        df = self._handle_missing_values(df, **handle_missing_kwargs)

        # Remove duplicates
        if remove_duplicates_kwargs is None:
            remove_duplicates_kwargs = {}
        else:
            # Map column names if specified
            if 'subset' in remove_duplicates_kwargs:
                remove_duplicates_kwargs['subset'] = map_columns(remove_duplicates_kwargs['subset'])
        df = self._remove_duplicates(df, **remove_duplicates_kwargs)

        # Normalize text columns
        if normalize_text_kwargs is None:
            normalize_text_kwargs = {}
        else:
            # Map column names if specified
            if 'columns' in normalize_text_kwargs:
                normalize_text_kwargs['columns'] = map_columns(normalize_text_kwargs['columns'])
        df = self._normalize_text_columns(df, **normalize_text_kwargs)

        # Normalize numeric columns (round decimals)
        if normalize_numeric_kwargs is None:
            normalize_numeric_kwargs = {}
        else:
            # Map column names if specified
            if 'columns' in normalize_numeric_kwargs:
                normalize_numeric_kwargs['columns'] = map_columns(normalize_numeric_kwargs['columns'])
        df = self._normalize_numeric_columns(df, **normalize_numeric_kwargs)

        # Convert date columns
        if convert_dates_kwargs is None:
            convert_dates_kwargs = {}
        else:
            # Map column names if specified
            if 'columns' in convert_dates_kwargs:
                convert_dates_kwargs['columns'] = map_columns(convert_dates_kwargs['columns'])
        df = self._convert_dates(df, **convert_dates_kwargs)

        # Convert categorical columns
        if convert_categories_kwargs is None:
            convert_categories_kwargs = {}
        else:
            # Map column names if specified
            if 'columns' in convert_categories_kwargs:
                convert_categories_kwargs['columns'] = map_columns(convert_categories_kwargs['columns'])
        df = self._convert_categories(df, **convert_categories_kwargs)

        if output_path is None:
            base = str(Path(file_path).with_suffix(''))
            output_path = f"{base}_cleaned.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned file saved to: {output_path}")
        return output_path

    def analyze_file(self, file_path: str, sheet_name: Union[str, int] = 0) -> dict:
        """
        Analyzes a CSV or Excel file and returns comprehensive insights about the data.
        
        This method provides detailed analysis including:
        - Basic information (shape, columns, data types)
        - Missing value counts
        - Duplicate row count
        - Numeric column statistics
        - Date column identification
        - Categorical column analysis
        - Sample data
        
        Args:
            file_path (str): Path to the input file (.csv, .xlsx, .xls)
            sheet_name (Union[str, int], optional): Sheet name or index to read for Excel files. 
                                                   Defaults to 0 (first sheet).
        
        Returns:
            dict: Dictionary containing comprehensive analysis results with the following keys:
                - shape: Tuple of (rows, columns)
                - columns: List of column names
                - dtypes: Dictionary mapping column names to data types
                - missing_values: Dictionary mapping column names to missing value counts
                - duplicate_rows: Number of duplicate rows
                - numeric_columns: List of numeric column names
                - numeric_summary: Dictionary with descriptive statistics for numeric columns
                - date_columns: List of date column names (including inferred ones)
                - category_columns: List of categorical column names
                - category_summary: Dictionary with value counts for categorical columns
                - sample: List of dictionaries representing first 10 rows of data
                
        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If the file format is not supported
        """
        logger.info(f"Analyzing file: {file_path}")
        df = self._read_file(file_path, sheet_name=sheet_name)
        # Add more detailed analysis for numbers, dates, categories
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        category_cols = df.select_dtypes(include=['category']).columns.tolist()
        # Try to infer possible date columns if not already datetime
        inferred_date_cols = []
        for col in df.columns:
            if "date" in col.lower() and col not in date_cols:
                try:
                    pd.to_datetime(df[col])
                    inferred_date_cols.append(col)
                except Exception:
                    pass
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.apply(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": numeric_cols,
            "numeric_summary": df[numeric_cols].describe().to_dict() if numeric_cols else {},
            "date_columns": date_cols + inferred_date_cols,
            "category_columns": category_cols,
            "category_summary": {col: df[col].value_counts(dropna=False).to_dict() for col in category_cols},
            "sample": df.head(10).to_dict(orient="records")
        }
        logger.info(f"Analysis: {analysis}")
        return analysis