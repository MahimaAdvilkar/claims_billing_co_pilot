import pandas as pd

class DataCoordinator:
    """
    Utility class for common data operations used by multiple agents.
    Keeps month ordering consistent across all agents.
    """

    MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    MONTH_MAP = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    @staticmethod
    def sort_by_month(df: pd.DataFrame, month_col: str = 'Service_Month') -> pd.DataFrame:
        """
        Sort dataframe by month (January to December order) and fill missing months with zeros.
        Ensures all months are present in the output, even if some have zero values.
        """
        months = DataCoordinator.MONTH_ORDER
        if month_col in df.columns:
            df_copy = df.copy()
            # If not already grouped, group and sum
            if df_copy.shape[0] < 12 or set(df_copy[month_col]) != set(months):
                df_copy = df_copy.groupby(month_col).sum(numeric_only=True).reset_index()
            df_copy = df_copy.set_index(month_col).reindex(months, fill_value=0).reset_index()
            return df_copy
        return df

    @staticmethod
    def get_month_number(month_name: str) -> int:
        """Get numeric value for month (1-12)"""
        return DataCoordinator.MONTH_MAP.get(month_name, 0)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> tuple:
        """
        Check if dataframe has all required columns.

        Returns:
            (is_valid: bool, missing_cols: list)
        """
        required_cols = [
            'Claim_ID', 'Service_Month', 'Amount_Billed', 'Amount_Received',
            'Status', 'Payer', 'Client_Name', 'Days_To_Payment',
            'Amount_Submitted', 'Denial_Reason'
        ]
        missing = [col for col in required_cols if col not in df.columns]
        return len(missing) == 0, missing