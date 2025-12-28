import pandas as pd
from .data_coordinator import DataCoordinator

class ClientHistoryAgent:
    """
    Identify repeat clients and rank by activity.
    Shows top 10 clients by total billed amount.
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        if df is None or len(df) == 0:
            print("⚠️ ClientHistoryAgent: No data available")
            return {"client_history": []}
        
        try:
            # Group by client
            client_data = df.groupby('Client_Name').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum',
                'Claim_ID': 'count',
                'Days_To_Payment': 'mean',
                'Service_Month': lambda x: ', '.join(sorted(set(x), 
                    key=lambda m: DataCoordinator.get_month_number(m)))
            }).reset_index()
            
            # Rename columns
            client_data.columns = ['Client_Name', 'Total_Billed', 'Total_Received', 
                                   'Claim_Count', 'Avg_Days_To_Payment', 'Months_Active']
            
            # Calculate metrics
            client_data['Variance'] = client_data['Total_Billed'] - client_data['Total_Received']
            client_data['Collection_Rate'] = (client_data['Total_Received'] / client_data['Total_Billed'] * 100).fillna(0).round(2)
            
            # Round currency
            for col in ['Total_Billed', 'Total_Received', 'Variance', 'Avg_Days_To_Payment']:
                client_data[col] = client_data[col].round(2)
            
            # Sort by total billed (most active first) and get top 10
            client_data = client_data.sort_values('Total_Billed', ascending=False).head(10)
            
            print(f"✅ Client history complete: {len(client_data)} top clients identified")
            return {"client_history": client_data.to_dict('records')}
            
        except Exception as e:
            print(f"❌ ClientHistoryAgent Error: {str(e)}")
            return {"client_history": []}