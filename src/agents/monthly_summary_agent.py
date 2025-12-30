import pandas as pd
from .data_coordinator import DataCoordinator

class MonthlySummaryAgent:
    """
    Generate monthly breakdown table for all 12 months.
    Shows billed, received, variance, and collection rate per month.
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        if df is None or len(df) == 0:
            print("⚠️ MonthlySummaryAgent: No data available")
            return {"monthly_summary": []}
        
        try:
            # Group by month
            monthly = df.groupby('Service_Month').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum',
                'Amount_Submitted': 'sum',
                'Claim_ID': 'count'
            }).reset_index()
            
            # Sort by month (Jan-Dec)
            monthly = DataCoordinator.sort_by_month(monthly, 'Service_Month')
            
            # Calculate derived metrics
            monthly['Variance'] = monthly['Amount_Billed'] - monthly['Amount_Received']
            monthly['Collection_Rate'] = (monthly['Amount_Received'] / monthly['Amount_Billed'] * 100).fillna(0).round(2)
            
            # Round currency columns
            for col in ['Amount_Billed', 'Amount_Received', 'Amount_Submitted', 'Variance']:
                monthly[col] = monthly[col].round(2)
            
            # Rename for display
            monthly.columns = ['Month', 'Billed', 'Received', 'Submitted', 'Claim_Count', 'Variance', 'Collection_Rate']
            
            print(f"✅ Monthly summary complete: {len(monthly)} months")
            return {"monthly_summary": monthly.to_dict('records')}
            
        except Exception as e:
            print(f"❌ MonthlySummaryAgent Error: {str(e)}")
            return {"monthly_summary": []}