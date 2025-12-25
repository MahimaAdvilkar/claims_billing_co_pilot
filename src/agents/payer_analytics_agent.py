import pandas as pd

class PayerAnalyticsAgent:
    """
    Deep dive into payer performance:
    - Total billed/received per payer
    - Collection rate per payer
    - Denial rate per payer
    - Average days to payment
    - Average claim value
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        if df is None or len(df) == 0:
            print("⚠️ PayerAnalyticsAgent: No data available")
            return {"payer_analytics": []}
        
        try:
            # Group by payer
            payer_analysis = df.groupby('Payer').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum',
                'Amount_Submitted': 'sum',
                'Claim_ID': 'count',
                'Days_To_Payment': 'mean'
            }).reset_index()
            
            payer_analysis.columns = ['Payer', 'Total_Billed', 'Total_Received', 
                                      'Total_Submitted', 'Claim_Count', 'Avg_Days_To_Payment']
            
            # Calculate derived metrics
            payer_analysis['Variance'] = payer_analysis['Total_Billed'] - payer_analysis['Total_Received']
            payer_analysis['Collection_Rate'] = (
                (payer_analysis['Total_Received'] / payer_analysis['Total_Billed'] * 100)
                .fillna(0).round(2)
            )
            
            # Denial count and rate per payer
            payer_analysis['Denial_Count'] = payer_analysis['Payer'].apply(
                lambda p: len(df[(df['Payer'] == p) & (df['Status'] == 'Denied')])
            )
            payer_analysis['Denial_Rate'] = (
                (payer_analysis['Denial_Count'] / payer_analysis['Claim_Count'] * 100)
                .fillna(0).round(2)
            )
            
            # Average claim value
            payer_analysis['Avg_Claim_Value'] = (
                (payer_analysis['Total_Billed'] / payer_analysis['Claim_Count'])
                .round(2)
            )
            
            # Round currency columns
            for col in ['Total_Billed', 'Total_Received', 'Total_Submitted', 
                       'Variance', 'Avg_Days_To_Payment', 'Avg_Claim_Value']:
                payer_analysis[col] = payer_analysis[col].round(2)
            
            # Sort by total billed (revenue)
            payer_analysis = payer_analysis.sort_values('Total_Billed', ascending=False)
            
            print(f"✅ Payer analytics complete: {len(payer_analysis)} payers analyzed")
            return {"payer_analytics": payer_analysis.to_dict('records')}
            
        except Exception as e:
            print(f"❌ PayerAnalyticsAgent Error: {str(e)}")
            return {"payer_analytics": []}