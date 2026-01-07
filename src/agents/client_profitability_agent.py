import pandas as pd

class ClientProfitabilityAgent:
    """
    Rank clients by profitability and risk.
    
    Risk Score calculation (0-100, lower = better):
    - 40% = (100 - Collection_Rate) → rewards high collection
    - 30% = (Days_To_Payment / 30 * 100) → penalizes slow payers
    - 30% = (Variance / Billed * 100) → penalizes high variance
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        if df is None or len(df) == 0:
            print("⚠️ ClientProfitabilityAgent: No data available")
            return {"client_profitability": []}
        
        try:
            # Group by client
            client_profit = df.groupby('Client_Name').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum',
                'Amount_Submitted': 'sum',
                'Claim_ID': 'count',
                'Days_To_Payment': 'mean'
            }).reset_index()
            
            client_profit.columns = ['Client', 'Total_Billed', 'Total_Received', 
                                     'Total_Submitted', 'Claim_Count', 'Avg_Days_To_Payment']
            
            # Profitability metrics
            client_profit['Variance'] = client_profit['Total_Billed'] - client_profit['Total_Received']
            client_profit['Collection_Rate'] = (
                (client_profit['Total_Received'] / client_profit['Total_Billed'] * 100)
                .fillna(0).round(2)
            )
            
            # Claims per month (on average)
            client_profit['Claim_Density'] = (
                (client_profit['Claim_Count'] / 12).round(2)
            )
            
            # RISK SCORE (lower = better, 0-100)
            # Component 1: Collection rate (40%)
            collection_component = (100 - client_profit['Collection_Rate']) * 0.4
            
            # Component 2: Payment speed (30%)
            payment_component = (client_profit['Avg_Days_To_Payment'] / 30 * 100) * 0.3
            
            # Component 3: Variance (30%)
            variance_component = (
                (client_profit['Variance'] / client_profit['Total_Billed'].abs() * 100)
                .fillna(0) * 0.3
            )
            
            client_profit['Risk_Score'] = (
                collection_component + payment_component + variance_component
            ).round(2)
            
            # Profitability rank (1 = most profitable)
            client_profit['Profit_Rank'] = client_profit['Collection_Rate'].rank(ascending=False).astype(int)
            
            # Round currency
            for col in ['Total_Billed', 'Total_Received', 'Total_Submitted', 
                       'Variance', 'Avg_Days_To_Payment']:
                client_profit[col] = client_profit[col].round(2)
            
            # Sort by risk score (lowest = safest/most profitable)
            client_profit = client_profit.sort_values('Risk_Score')
            
            print(f"✅ Profitability analysis complete: {len(client_profit)} clients ranked")
            return {"client_profitability": client_profit.to_dict('records')}
            
        except Exception as e:
            print(f"❌ ClientProfitabilityAgent Error: {str(e)}")
            return {"client_profitability": []}