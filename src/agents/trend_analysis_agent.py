import pandas as pd

class TrendAnalysisAgent:
    """Analyze monthly trends and growth patterns"""
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        if df is None:
            print("⚠️ TrendAnalysisAgent: No data available")
            return {"trend_analysis": {}}
        
        # Convert Service_Month to numeric (Jan=1, Dec=12)
        month_order = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['Month_Num'] = df['Service_Month'].map(month_order)
        
        # Monthly aggregation
        monthly_trend = df.groupby(['Month_Num', 'Service_Month']).agg({
            'Amount_Billed': 'sum',
            'Amount_Received': 'sum',
            'Amount_Submitted': 'sum',
            'Claim_ID': 'count',
            'Days_To_Payment': 'mean'
        }).reset_index().sort_values('Month_Num')
        
        monthly_trend.columns = ['Month_Num', 'Month', 'Billed', 'Received', 'Submitted', 'Claim_Count', 'Avg_Days']
        
        # Calculate month-over-month changes
        monthly_trend['Billed_Change'] = monthly_trend['Billed'].diff()
        monthly_trend['Billed_Change_Pct'] = (monthly_trend['Billed'].pct_change() * 100).round(2)
        monthly_trend['Received_Change'] = monthly_trend['Received'].diff()
        monthly_trend['Received_Change_Pct'] = (monthly_trend['Received'].pct_change() * 100).round(2)
        monthly_trend['Collection_Rate'] = (monthly_trend['Received'] / monthly_trend['Billed'] * 100).round(2)
        
        # Round currency
        for col in ['Billed', 'Received', 'Submitted', 'Avg_Days', 'Billed_Change', 'Received_Change']:
            monthly_trend[col] = monthly_trend[col].round(2)
        
        # Growth summary
        first_month_billed = monthly_trend['Billed'].iloc[0]
        last_month_billed = monthly_trend['Billed'].iloc[-1]
        year_growth_pct = ((last_month_billed - first_month_billed) / first_month_billed * 100).round(2)
        
        trend_analysis = {
            "monthly_data": monthly_trend.to_dict('records'),
            "year_growth_pct": float(year_growth_pct),
            "highest_month": {
                "month": monthly_trend.loc[monthly_trend['Billed'].idxmax(), 'Month'],
                "amount": float(monthly_trend['Billed'].max())
            },
            "lowest_month": {
                "month": monthly_trend.loc[monthly_trend['Billed'].idxmin(), 'Month'],
                "amount": float(monthly_trend['Billed'].min())
            }
        }
        return {"trend_analysis": trend_analysis}