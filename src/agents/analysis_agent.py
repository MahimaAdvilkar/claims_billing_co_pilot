import pandas as pd
import re

class AnalysisAgent:
    """
    Calculate overall summary metrics from master dataframe.
    Detects month filters from user query.
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        # Check if data loaded successfully
        if df is None or len(df) == 0:
            print("⚠️ AnalysisAgent: No data available")
            return {"analysis": {}}
        
        try:
            query = state.get("query", "").lower()
            
            # Try to extract month from query
            month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']
            
            filtered_df = df
            filtered_month = ""
            for month in month_names:
                if month in query:
                    filtered_df = df[df['Service_Month'] == month.capitalize()]
                    filtered_month = month.capitalize()
                    print(f"📅 Filtered to {month.capitalize()}: {len(filtered_df)} claims")
                    break
            
            # Calculate core metrics
            total_billed = float(filtered_df['Amount_Billed'].sum())
            total_received = float(filtered_df['Amount_Received'].sum())
            total_variance = total_billed - total_received
            
            collection_rate = (total_received / total_billed * 100) if total_billed > 0 else 0
            
            # Claim counts by status
            total_claims = len(filtered_df)
            paid_claims = len(filtered_df[filtered_df['Status'] == 'Paid'])
            denied_claims = len(filtered_df[filtered_df['Status'] == 'Denied'])
            partial_claims = len(filtered_df[filtered_df['Status'] == 'Partial'])
            
            # Breakdowns for display
            by_status = filtered_df.groupby('Status').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum',
                'Claim_ID': 'count'
            }).to_dict('index')
            
            by_payer = filtered_df.groupby('Payer').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum',
                'Claim_ID': 'count'
            }).to_dict('index')
            
            analysis = {
                "total_billed": total_billed,
                "total_received": total_received,
                "total_variance": total_variance,
                "collection_rate": float(collection_rate),
                "total_claims": int(total_claims),
                "paid_claims": int(paid_claims),
                "denied_claims": int(denied_claims),
                "partial_claims": int(partial_claims),
                "by_status": by_status,
                "by_payer": by_payer
            }
            
            print(f"✅ Analysis complete: ${total_billed:,.0f} billed, {collection_rate:.1f}% collection rate")
            # Only return keys this agent is responsible for
            result = {"analysis": analysis}
            if filtered_month:
                result["filtered_month"] = filtered_month
            return result
            
        except Exception as e:
            print(f"❌ AnalysisAgent Error: {str(e)}")
            return {"analysis": {}}